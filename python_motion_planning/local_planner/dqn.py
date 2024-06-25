"""
@file: ddpg.py
@breif: Deep Deterministic Policy Gradient (DDPG) motion planning.
@author: Wu Maojia
@update: 2024.5.24
"""
import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import random
from tqdm import tqdm
import math
import copy
import datetime
import os
from collections import namedtuple, deque

from .local_planner import LocalPlanner
from python_motion_planning.utils import Env, MathHelper, Robot

ActionRot = namedtuple("ActionRot", ["v", "w"])

class BasicBuffer:
    '''
    * @breif: 基础经验回放池
    '''
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    '''
    * @breif: 向经验池推入一条经验
    '''
    def push(self, *experience):
        state, action, reward, next_state, done = experience
        self.buffer.append((state, action, np.array([reward]), next_state, done))

    '''
    * @breif: 采样一个batch的数据
    '''
    def sample(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = [], [], [], [], []
        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    '''
    * @breif: 采样一个batch的数据, 且这个batch是连续的
    '''
    def sampleSequence(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = [], [], [], [], []
        start = np.random.randint(0, len(self.buffer) - batch_size)

        for sample in range(start, start + batch_size):
            state, action, reward, next_state, done = self.buffer[sample]
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def __len__(self):
        return len(self.buffer)

class SumTree:
    '''
    * @breif: 求和树
    * @attention: 容量只能为偶数
    '''
    def __init__(self, capacity):
        # 求和树容量
        self.capacity = capacity
        # 树结构
        self.tree = np.zeros(2 * capacity - 1)
        # 树叶节点
        self.data = np.zeros(capacity, dtype=object)
        # 指向当前树叶节点的指针
        self.write = 0
        # 求和树缓存的数据量
        self.size = 0

    '''
    * @breif: 递归更新树的优先级
    * @param[in]: idx   ->  索引
    * @param[in]: change->  优先级增量
    * @example: 六节点求和树的索引
    *               0
    *              / \
    *             1   2
    *            / \ / \
    *           3  4 5  6
    *          / \ / \
    *         7  8 9 10
    '''    
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    '''
    * @breif: 递归求叶节点(s落在某个节点区间内)
    * @param[in]: idx   ->  子树根节点索引
    * @param[in]: s     ->  采样优先级
    '''    
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    '''
    * @breif: 返回根节点, 即总优先级权重
    '''    
    def total(self):
        return self.tree[0]

    '''
    * @breif: 添加带优先级的数据到求和树
    * @param[in]: p   ->  优先级
    * @param[in]: data->  数据
    '''    
    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        self.size = min(self.capacity, self.size + 1)
        if self.write >= self.capacity:
            self.write = 0

    '''
    * @breif: 更新求和树数据
    * @param[in]: idx   ->  索引
    * @param[in]: p   ->  优先级
    '''    
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
        self.tree = self.tree / self.tree.max()

    '''
    * @breif: 根据采样值求叶节点数据
    * @param[in]: s     ->  采样优先级
    '''    
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedBuffer:
    '''
    * @breif: 优先级经验回放池
    '''
    def __init__(self, max_size, alpha=0.6, beta=0.4):
        self.sum_tree = SumTree(max_size)
        self.alpha = alpha
        self.beta = beta
        self.cur_size = 0

    '''
    * @breif: 向经验池推入一条经验
    '''    
    def push(self, *experience):
        priority = 1.0 if self.cur_size == 0 else self.sum_tree.tree.max()
        self.cur_size = self.cur_size + 1
        state, action, reward, next_state, done = experience
        self.sum_tree.add(priority, (state, action, np.array([reward]), next_state, done))

    '''
    * @breif: 采样一个batch的数据
    '''
    def sample(self, batch_size):
        batch_idx, batch, probs = [], [], []
        segment = self.sum_tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.sum_tree.get(s)
            batch_idx.append(idx)
            batch.append(data)
            probs.append(p)

        weights = np.power(self.sum_tree.size * np.array(probs) / self.sum_tree.total(), -self.beta)
        weights = (weights / weights.max()).tolist()

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = [], [], [], [], []
        for transition in batch:
            state, action, reward, next_state, done = transition
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch), batch_idx, weights

    '''
    * @breif: 根据时序差分误差更新经验池
    '''
    def updatePriority(self, idx, td_error):
        priority = td_error ** self.alpha
        self.sum_tree.update(idx, priority)

    def __len__(self):
        return self.cur_size


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.fc = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim)
        )

    def forward(self, state):
        qvals = self.fc(state)
        return qvals

class DQNPlanner(LocalPlanner):
    """
    Class for Fully Connected Deep Q-Value Network (DQN) motion planning.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Env): environment
        heuristic_type (str): heuristic function type
        hidden_depth (int): the number of hidden layers of the neural network
        hidden_width (int): the number of neurons in hidden layers of the neural network
        batch_size (int): batch size to optimize the neural networks
        buffer_size (int): maximum replay buffer size
        gamma (float): discount factor
        tau (float): Softly update the target network
        lr (float): learning rate
        train_noise (float): Action noise coefficient during training for exploration
        random_episodes (int): Take the random actions in the beginning for the better exploration
        max_episode_steps (int): Maximum steps for each episode
        update_freq (int): Frequency (times) of updating the network for each step
        update_steps (int): Update the network for every 'update_steps' steps
        evaluate_freq (int): Frequency (times) of evaluations and calculate the average
        evaluate_episodes (int): Evaluate the network every 'evaluate_episodes' episodes
        actor_save_path (str): Save path of the trained actor network
        critic_save_path (str): Save path of the trained critic network
        actor_load_path (str): Load path of the trained actor network
        critic_load_path (str): Load path of the trained critic network
        **params: other parameters can be found in the parent class LocalPlanner

    Examples:
        >>> from python_motion_planning.utils import Grid
        >>> from python_motion_planning.local_planner import DDPG
        >>> plt = DDPG(start=(5, 5, 0), goal=(45, 25, 0), env=Grid(51, 31),
            actor_save_path="models/actor_best.pth", critic_save_path="models/critic_best.pth")
        >>> plt.train(num_episodes=10000)
        
        # load the trained model and run
        >>> plt = DDPG(start=(5, 5, 0), goal=(45, 25, 0), env=Grid(51, 31),
            actor_load_path="models/actor_best.pth", critic_load_path="models/critic_best.pth")
        >>> plt.run()
    """
    def __init__(self, start: tuple, goal: tuple, env: Env, heuristic_type: str = "euclidean",
                 batch_size: int = 2000, buffer_size: int = 1e6,
                 gamma: float = 0.999, tau: float = 1e-3, lr: float = 1e-4, train_noise: float = 0.1,
                 random_episodes: int = 50, max_episode_steps: int = 200,
                 update_freq: int = 1, update_steps: int = 1, evaluate_freq: int = 50, evaluate_episodes: int = 50,
                 model_save_path: str = "models/dqn_best.pth",
                 model_load_path: str = None,
                 **params) -> None:
        super().__init__(start, goal, env, heuristic_type, **params)
        # DDPG parameters
        self.batch_size = int(batch_size)       # batch size to optimize the neural networks
        self.buffer_size = int(buffer_size)     # maximum replay buffer size
        self.gamma = gamma                      # discount factor
        self.tau = tau                          # Softly update the target network
        self.lr = lr                            # learning rate
        self.train_noise = train_noise          # Action noise coefficient during training for exploration
        self.random_episodes = random_episodes  # Take the random actions in the beginning for the better exploration
        self.max_episode_steps = max_episode_steps  # Maximum steps for each episode
        self.update_freq = update_freq          # Frequency (times) of updating the network for each step
        self.update_steps = update_steps        # Update the network for every 'update_steps' steps
        self.evaluate_freq = evaluate_freq      # Frequency (times) of evaluations and calculate the average
        self.evaluate_episodes = evaluate_episodes      # Evaluate the network every 'evaluate_episodes' episodes
        self.model_save_path = model_save_path      # Save path of the trained network
        self.epsilon, self.epsilon_max, self.epsilon_delta = 0.0, 0.95, 5e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.action_space = self.buildActionSpace()
        self.n_observations = 8     # x, y, theta, v, w, g_x, g_y, g_theta
        self.n_actions = len(self.action_space) 

        self.model = DQN(self.n_observations, self.n_actions).to(self.device)
        if model_load_path:
            self.model.load_state_dict(torch.load(model_load_path))
        self.target_model = DQN(self.n_observations, self.n_actions).to(self.device)
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(param)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

        self.criterion = nn.MSELoss()

        # self.replay_buffer = BasicBuffer(max_size=self.buffer_size)
        self.replay_buffer = PrioritizedBuffer(max_size=self.buffer_size)

        # Build a tensorboard
        self.writer = SummaryWriter(log_dir='runs/DQN_{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

        # global planner
        g_start = (start[0], start[1])
        g_goal  = (goal[0], goal[1])
        self.g_planner = {"planner_name": "a_star", "start": g_start, "goal": g_goal, "env": env}
        self.path = self.g_path[::-1]

    def __del__(self) -> None:
        self.writer.close()

    def __str__(self) -> str:
        return "Fully Connected Deep Q-Value Network (DQN)"

    def plan(self) -> tuple:
        """
        Deep Deterministic Policy Gradient (DDPG) motion plan function.

        Returns:
            flag (bool): planning successful if true else failed
            pose_list (list): history poses of robot
        """
        s = self.reset()
        for _ in range(self.params["MAX_ITERATION"]):
            # break until goal reached
            if self.reach_goal(tuple(s[0:3]), tuple(s[5:8])):
                return True, self.robot.history_pose

            # get the particular point on the path at the lookahead distance to track
            lookahead_pt, theta_trj, kappa = self.getLookaheadPoint()
            s[5:7] = torch.tensor(lookahead_pt, device=self.device)
            s[7] = torch.tensor(theta_trj, device=self.device)

            a = self.policy(s)   # get the action from the actor network
            s_, r, done, win = self.step(s, a)  # take the action and get the next state and reward
            s = s_  # Move to the next state
            self.robot.px, self.robot.py, self.robot.theta, self.robot.v, self.robot.w = tuple(s[0:5].cpu().numpy())

        return True, self.robot.history_pose
        # return False, None

    def run(self) -> None:
        """
        Running both plannig and animation.
        """
        _, history_pose = self.plan()
        print(f"Number of iterations: {len(history_pose)}")
        if not history_pose:
            raise ValueError("Path not found and planning failed!")

        path = np.array(history_pose)[:, 0:2]
        cost = np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1, keepdims=True)))
        self.plot.plotPath(self.path, path_color="r", path_style="--")
        self.plot.animation(path, str(self), cost, history_pose=history_pose)

    def buildActionSpace(self):
        '''
        Action space consists of 25 uniformly sampled actions in permitted range
        and 25 randomly sampled actions.
        '''
        speed_samples, rotation_samples = 5, 16
        speeds = [
            (np.exp((i + 1) / speed_samples) - 1) / (np.e - 1) * self.params["MAX_V"]
            for i in range(speed_samples)
        ]
        rotations = np.linspace(self.params["MIN_W"], self.params["MAX_W"], rotation_samples)

        action_space = [ActionRot(0, 0)]
        for rotation, speed in itertools.product(rotations, speeds):
            action_space.append(ActionRot(speed, rotation))

        return action_space

    def optimize_model(self) -> tuple:
        """
        Optimize the neural networks when training.

        Returns:
            actor_loss (float): actor loss
            critic_loss (float): critic loss
        """
        # basic buffer
        # states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # priority buffer
        transitions, idxs, weights = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = transitions

        states = torch.stack(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = (1 - torch.FloatTensor(dones)).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # basic buffer
        # curr_Q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # next_Q = self.target_model(next_states)
        # max_next_Q = torch.max(next_Q, 1)[0]
        # expected_Q = rewards.squeeze(1) + self.gamma * max_next_Q * dones
        # loss = self.criterion(curr_Q, expected_Q.detach())

        # priority buffer
        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_a = torch.argmax(self.model.forward(next_states), dim=1)
        next_Q = self.target_model.forward(next_states).gather(1, next_a.unsqueeze(1)).squeeze(1)
        expected_Q = rewards.squeeze(1) + self.gamma * next_Q * dones

        td_errors = torch.abs(curr_Q - expected_Q)
        loss = self.criterion(torch.sqrt(weights) * curr_Q, torch.sqrt(weights) * expected_Q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新target网络
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        self.epsilon = self.epsilon + self.epsilon_delta \
                if self.epsilon < self.epsilon_max else self.epsilon_max
        
        # 根据时序差分更新优先级
        for idx, td_error in zip(idxs, td_errors.cpu().detach().numpy()):
            self.replay_buffer.updatePriority(idx, td_error)

        return loss.item()

    def evaluate_policy(self) -> float:
        """
        Evaluate the policy and calculating the average reward.

        Returns:
            evaluate_reward (float): average reward of the policy
        """
        print(f"Evaluating: ")
        evaluate_reward = 0
        for _ in tqdm(range(self.evaluate_freq)):
            s = self.reset(random_sg=True)
            done = False
            episode_reward = 0
            step = 0
            while not done:
                a = self.policy(s)  # We do not add noise when evaluating
                s_, r, done, win = self.step(s, a)
                self.replay_buffer.push(s, a, r, s_, win)  # Store the transition
                episode_reward += r
                s = s_
                step += 1
                if step >= self.max_episode_steps:
                    break
            evaluate_reward += episode_reward / step

        return evaluate_reward / self.evaluate_freq

    def policy(self, state, mode='random'):
        # state = torch.FloatTensor(state).unsqueeze(0).to(self.device)   # 化成batch_size=1的数据
        qvals = self.model.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())
        if mode=='random' and np.random.randn() > self.epsilon:
            return np.random.randint(0, len(self.action_space))
        return action

    def train(self, num_episodes: int = 10000) -> None:
        """
        Train the model.

        Parameters:
            num_episodes (int): number of episodes to train the model
        """
        noise_std = self.train_noise * torch.tensor([
            self.params["MAX_V_INC"] - self.params["MIN_V_INC"],
            self.params["MAX_W_INC"] - self.params["MIN_W_INC"]
        ], device=self.device)  # the std of Gaussian noise for exploration

        best_reward = -float('inf')

        # Train the model
        for episode in range(1, num_episodes+1):
            print(f"Episode: {episode}/{num_episodes}, Training: ")
            s = self.reset(random_sg=True)
            episode_loss = 0
            optimize_times = 0
            for episode_steps in tqdm(range(1, self.max_episode_steps+1)):
                a = self.policy(s)
                s_, r, done, win = self.step(s, a)
                self.replay_buffer.push(s, a, r, s_, win)  # Store the transition

                # update the networks if enough samples are available
                if episode > self.random_episodes and (episode_steps % self.update_steps == 0 or done):
                    for _ in range(self.update_freq):
                        loss = self.optimize_model()
                        episode_loss += loss
                        optimize_times += 1

                if win:
                    print(f"Goal reached! State: {s}, Action: {a}, Reward: {r:.4f}, Next State: {s_}")
                    break
                elif done:  # lose (collide)
                    print(f"Collision! State: {s}, Action: {a}, Reward: {r:.4f}, Next State: {s_}")
                    break

                s = s_  # Move to the next state

            if episode > self.random_episodes:
                average_loss = episode_loss / optimize_times
                self.writer.add_scalar('train loss', average_loss, global_step=episode)

            if episode % self.evaluate_episodes == 0 and episode > self.random_episodes - self.evaluate_episodes:
                print()
                evaluate_reward = self.evaluate_policy()
                print("Evaluate_reward:{}".format(evaluate_reward))
                print()
                self.writer.add_scalar('Evaluate reward', evaluate_reward, global_step=episode)

                # Save the model
                if evaluate_reward > best_reward:
                    best_reward = evaluate_reward

                    # Create the directory if it does not exist
                    if not os.path.exists(os.path.dirname(self.model_save_path)):
                        os.makedirs(os.path.dirname(self.model_save_path))

                    torch.save(self.model.state_dict(), self.model_save_path)

    def reset(self, random_sg: bool = False) -> torch.Tensor:
        """
        Reset the environment and the robot.

        Parameters:
            random_sg (bool): whether to generate random start and goal or not

        Returns:
            state (torch.Tensor): initial state of the robot
        """
        if random_sg:   # random start and goal
            start = (random.uniform(0, self.env.x_range), random.uniform(0, self.env.y_range), random.uniform(-math.pi, math.pi))
            # generate random start and goal until they are not in collision
            while self.in_collision(start):
                start = (random.uniform(0, self.env.x_range), random.uniform(0, self.env.y_range), random.uniform(-math.pi, math.pi))

            # goal is on the circle with radius self.params["MAX_LOOKAHEAD_DIST"] centered at start
            goal_angle = random.uniform(-math.pi, math.pi)
            goal_dist = self.params["MAX_LOOKAHEAD_DIST"]
            goal_x = start[0] + goal_dist * math.cos(goal_angle)
            goal_y = start[1] + goal_dist * math.sin(goal_angle)
            goal = (goal_x, goal_y, goal_angle)

            while self.in_collision(goal):
                goal_angle = random.uniform(-math.pi, math.pi)
                goal_dist = self.params["MAX_LOOKAHEAD_DIST"]
                goal_x = start[0] + goal_dist * math.cos(goal_angle)
                goal_y = start[1] + goal_dist * math.sin(goal_angle)
                goal = (goal_x, goal_y, goal_angle)

        else:
            start = self.start
            goal = self.goal

        self.robot = Robot(start[0], start[1], start[2], 0, 0)
        state = self.robot.state    # np.array([[self.px], [self.py], [self.theta], [self.v], [self.w]])
        state = np.pad(state, pad_width=((0, 3), (0, 0)), mode='constant')
        state[5:8, 0] = goal
        state = torch.tensor(state, device=self.device, dtype=torch.float).squeeze(dim=1)
        return state

    def step(self, state: torch.Tensor, action) -> tuple:
        """
        Take a step in the environment.

        Parameters:
            state (torch.Tensor): current state of the robot
            action (torch.Tensor): action to take

        Returns:
            next_state (torch.Tensor): next state of the robot
            reward (float): reward for taking the action
            done (bool): whether the episode is done
        """
        dt = self.params["TIME_STEP"]
        v_d = state[3].item() + self.action_space[action].v * dt
        w_d = state[4].item() + self.action_space[action].w * dt
        self.robot.kinematic(np.array([[v_d], [w_d]]), dt)
        next_state = self.robot.state
        next_state = np.pad(next_state, pad_width=((0, 3), (0, 0)), mode='constant')
        next_state = torch.tensor(next_state, device=self.device, dtype=torch.float).squeeze(dim=1)
        next_state[5:8] = state[5:8]
        next_state[2] = self.regularizeAngle(next_state[2].item())
        next_state[3] = MathHelper.clamp(next_state[3].item(), self.params["MIN_V"], self.params["MAX_V"])
        next_state[4] = MathHelper.clamp(next_state[4].item(), self.params["MIN_W"], self.params["MAX_W"])
        win = self.reach_goal(tuple(next_state[0:3]), tuple(next_state[5:8]))
        lose = self.in_collision(tuple(next_state[0:2]))
        reward = self.reward(state, next_state, win, lose)
        done = win or lose
        return next_state, reward, done, win

    def reward(self, prev_state: torch.Tensor, state: torch.Tensor, win: bool, lose: bool) -> float:
        """
        The state reward function.

        Parameters:
            state (torch.Tensor): current state of the robot
            win (bool): whether the episode is won (reached the goal)
            lose (bool): whether the episode is lost (collided)

        Returns:
            reward (float): reward for the current state
        """
        reward = 0

        goal_dist = self.dist((state[0], state[1]), (state[5], state[6]))
        scaled_goal_dist = goal_dist / self.params["MAX_LOOKAHEAD_DIST"]

        reward -= scaled_goal_dist

        prev_goal_dist = self.dist((prev_state[0], prev_state[1]), (state[5], state[6]))
        if goal_dist < prev_goal_dist:
            reward += 0.01
        else:
            reward -= 0.01

        if win:
            reward += self.max_episode_steps
        if lose:
            reward -= self.max_episode_steps / 5.0

        return reward
