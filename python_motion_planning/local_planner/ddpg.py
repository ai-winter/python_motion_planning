"""
@file: ddpg.py
@breif: Deep Deterministic Policy Gradient (DDPG) motion planning.
@author: Wu Maojia
@update: 2024.5.19
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm import tqdm
import math
import copy
import datetime
import os
import heapq
from scipy.spatial.distance import cdist
from torch.utils.tensorboard import SummaryWriter

from .local_planner import LocalPlanner
from python_motion_planning.utils import Env, MathHelper, Robot


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device):
        self.max_size = int(1e6)
        self.count = 0
        self.size = 0
        self.s = torch.zeros((self.max_size, state_dim), dtype=torch.float, device=device)   # state
        self.a = torch.zeros((self.max_size, action_dim), dtype=torch.float, device=device)  # action
        self.r = torch.zeros((self.max_size, 1), dtype=torch.float, device=device)           # reward
        self.s_ = torch.zeros((self.max_size, state_dim), dtype=torch.float, device=device)  # next state
        self.dw = torch.zeros((self.max_size, 1), dtype=torch.bool, device=device)           # dead or win, True: win (done), False: dead (not done).

    def store(self, s, a, r, s_, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = torch.tensor(dw, dtype=torch.bool)
        self.count = (self.count + 1) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(self.size + 1, self.max_size)  # Record the number of  transitions

    def sample(self, batch_size):
        index = torch.randint(self.size, size=(batch_size,))  # Randomly sampling
        batch_s = self.s[index]
        batch_a = self.a[index]
        batch_r = self.r[index]
        batch_s_ = self.s_[index]
        batch_dw = self.dw[index]

        return batch_s, batch_a, batch_r, batch_s_, batch_dw


# class PrioritizedReplayBuffer(object):
#     def __init__(self, state_dim, action_dim, device, alpha=0.6, beta_start=0.4, beta_frames=100000):
#         self.max_size = int(1e6)
#         self.count = 0
#         self.size = 0
#         self.alpha = alpha  # Importance sampling parameter
#         self.beta_start = beta_start
#         self.beta_frames = beta_frames
#         self.frame = 1
#         self.beta = self.beta_schedule(self.frame)
#         self.device = device
#
#         self.s = torch.zeros((self.max_size, state_dim), dtype=torch.float, device=self.device)  # state
#         self.a = torch.zeros((self.max_size, action_dim), dtype=torch.float, device=self.device)  # action
#         self.r = torch.zeros((self.max_size, 1), dtype=torch.float, device=self.device)  # reward
#         self.s_ = torch.zeros((self.max_size, state_dim), dtype=torch.float, device=self.device)  # next state
#         self.dw = torch.zeros((self.max_size, 1), dtype=torch.bool, device=self.device)  # dead or win
#         self.priorities = torch.zeros((self.max_size, 1), dtype=torch.float, device=self.device)  # priorities initialized to 1.0
#
#         self.heap = []  # Min-heap for efficient sampling
#
#     def beta_schedule(self, frame_idx):
#         return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
#
#     def store(self, s, a, r, s_, dw):
#         idx = self.count % self.max_size
#         self.s[idx] = s
#         self.a[idx] = a
#         self.r[idx] = r
#         self.s_[idx] = s_
#         self.dw[idx] = torch.tensor(dw, dtype=torch.bool)
#         self.priorities[idx] = max(self.priorities.max().item(), 1e-6)  # Set minimum priority to 1e-6 to avoid division by zero errors
#         entry = (-self.priorities[idx], idx)  # Min heap, so we use negative priorities
#         heapq.heappush(self.heap, entry)
#         self.count = (self.count + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)
#
#     def sample(self, batch_size):
#         if self.size < batch_size:
#             raise ValueError("Not enough samples in buffer yet.")
#
#         # Sample indices based on priorities
#         segment = self.size // batch_size
#         indices = [heapq.heappop(self.heap)[1] for _ in range(batch_size)]
#         batch_s = self.s[indices]
#         batch_a = self.a[indices]
#         batch_r = self.r[indices]
#         batch_s_ = self.s_[indices]
#         batch_dw = self.dw[indices]
#         priorities = self.priorities[indices]
#
#         # Importance Sampling weights
#         weights = (self.size * priorities) ** -self.alpha
#         weights /= weights.max()
#         weights = torch.tensor(weights, dtype=torch.float, device=self.device)
#         weights *= (self.beta_start ** self.frame)  # Annealing beta
#
#         self.frame += 1
#         self.beta = self.beta_schedule(self.frame)
#
#         return batch_s, batch_a, batch_r, batch_s_, batch_dw, indices, weights
#
#     def update_priorities(self, indices, new_priorities):
#         for idx, priority in zip(indices, new_priorities):
#             self.priorities[idx] = priority
#             # Update the heap
#             heapq.heapify(
#                 self.heap)  # This is inefficient; ideally, only the affected entries would be updated in-place


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_depth, hidden_width, min_state, max_state, min_action, max_action):
        super(Actor, self).__init__()
        self.min_state = min_state
        self.max_state = max_state
        self.min_action = min_action
        self.max_action = max_action

        self.hidden_depth = hidden_depth
        self.input_layer = nn.Linear(state_dim, hidden_width)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_width, hidden_width) for _ in range(self.hidden_depth)])
        self.output_layer = nn.Linear(hidden_width, action_dim)

    def forward(self, s):
        # 归一化
        s = (s - self.min_state) / (self.max_state - self.min_state)

        s = F.relu(self.input_layer(s))
        for i in range(self.hidden_depth):
            s = F.relu(self.hidden_layers[i](s))
        s = self.output_layer(s)
        a = self.min_action + (self.max_action - self.min_action) * torch.sigmoid(s)  # [min,max]
        return a


class Critic(nn.Module):  # According to (s,a), directly calculate q(s,a)
    def __init__(self, state_dim, action_dim, hidden_depth, hidden_width, min_state, max_state, min_action, max_action):
        super(Critic, self).__init__()
        self.min_state = min_state
        self.max_state = max_state
        self.min_action = min_action
        self.max_action = max_action

        self.hidden_depth = hidden_depth
        self.input_layer = nn.Linear(state_dim, hidden_width)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_width, hidden_width) for _ in range(self.hidden_depth)])
        self.output_layer = nn.Linear(hidden_width, action_dim)

    def forward(self, s, a):
        # 归一化
        s = (s - self.min_state) / (self.max_state - self.min_state)
        a = (a - self.min_action) / (self.max_action - self.min_action)

        input = torch.cat([s, a], axis=-1)

        q = F.relu(self.input_layer(input))
        for i in range(self.hidden_depth):
            q = F.relu(self.hidden_layers[i](q))
        q = self.output_layer(q)
        return q


class DDPG(LocalPlanner):
    """
    Class for Deep Deterministic Policy Gradient (DDPG) motion planning.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Env): environment
        heuristic_type (str): heuristic function type

    Examples:
        >>> from python_motion_planning.utils import Grid
        >>> from python_motion_planning.local_planner import DDPG
        >>> start = (5, 5, 0)
        >>> goal = (45, 25, 0)
        >>> env = Grid(51, 31)
        >>> planner = DDPG(start, goal, env)
        >>> planner.run()

    References:
        [1] Continuous control with deep reinforcement learning
    """
    def __init__(self, start: tuple, goal: tuple, env: Env, heuristic_type: str = "euclidean",
                 hidden_depth=2, hidden_width=512, batch_size=100, gamma=0.999, tau=1e-3, lr=5e-4, random_episodes=30,
                 update_freq=1, evaluate_times=30, evaluate_freq=30,
                 actor_save_path="models/actor_best.pth",
                 critic_save_path="models/critic_best.pth",
                 actor_load_path=None,
                 critic_load_path=None, **params) -> None:
        super().__init__(start, goal, env, heuristic_type, **params)
        # DDPG parameters
        self.hidden_depth = hidden_depth        # The number of hidden layers of the neural network
        self.hidden_width = hidden_width        # The number of neurons in hidden layers of the neural network
        self.batch_size = batch_size            # batch size
        self.gamma = gamma                      # discount factor
        self.tau = tau                          # Softly update the target network
        self.lr = lr                            # learning rate
        self.random_episodes = random_episodes  # Take the random actions in the beginning for the better exploration
        self.update_freq = update_freq          # Update the network every 'update_freq' steps if episode > exploration_episodes
        self.evaluate_times = evaluate_times    # Times of evaluations and calculate the average
        self.evaluate_freq = evaluate_freq      # Evaluate the network every 'evaluate_freq' episodes
        # self.win_reward = 1.0   # Reward for winning the game (reach the goal)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.actor_save_path = actor_save_path
        self.critic_save_path = critic_save_path

        self.n_observations = 9     # x, y, theta, v, w, g_x, g_y, g_theta, steps
        self.n_actions = 2          # v_inc, w_inc

        self.min_state = torch.tensor([0, 0, -math.pi, self.params["MIN_V"], self.params["MIN_W"], 0, 0, -math.pi, 0],
                                      device=self.device)
        self.max_state = torch.tensor([self.env.x_range, self.env.y_range, math.pi, self.params["MAX_V"],
                                       self.params["MAX_W"], self.env.x_range, self.env.y_range, math.pi,
                                       self.params["MAX_ITERATION"]], device=self.device)
        self.min_action = torch.tensor([self.params["MIN_V_INC"], self.params["MIN_W_INC"]], device=self.device)
        self.max_action = torch.tensor([self.params["MAX_V_INC"], self.params["MAX_W_INC"]], device=self.device)

        self.actor = Actor(self.n_observations, self.n_actions, self.hidden_depth, self.hidden_width, self.min_state,
                           self.max_state, self.min_action, self.max_action).to(self.device)
        if actor_load_path:
            self.actor.load_state_dict(torch.load(actor_load_path))
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(self.n_observations + self.n_actions, 1, self.hidden_depth, self.hidden_width,
                             self.min_state, self.max_state, self.min_action, self.max_action).to(self.device)
        if critic_load_path:
            self.critic.load_state_dict(torch.load(critic_load_path))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, weight_decay=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr, weight_decay=1e-4)

        self.criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(self.n_observations, self.n_actions, device=self.device)
        # self.replay_buffer = PrioritizedReplayBuffer(self.n_observations, self.n_actions, device=self.device)

        self.total_reward = 0   # accumulate reward in each episode

        # Build a tensorboard
        self.writer = SummaryWriter(log_dir='runs/DDPG_{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

        # # global planner
        # g_start = (start[0], start[1])
        # g_goal  = (goal[0], goal[1])
        # self.g_planner = {"planner_name": "a_star", "start": g_start, "goal": g_goal, "env": env}
        # self.path = self.g_path[::-1]

    def __del__(self):
        self.writer.close()

    def __str__(self) -> str:
        return "Deep Deterministic Policy Gradient (DDPG)"

    def plan(self):
        """
        Deep Deterministic Policy Gradient (DDPG) motion plan function.

        Returns:
            flag (bool): planning successful if true else failed
            pose_list (list): history poses of robot
        """
        s = self.reset()
        episode_reward = 0
        for step in range(self.params["MAX_ITERATION"]):
            a = self.select_action(s)
            s_, r, done, win = self.step(s, a)
            if done:
                print(f"Step: {step}, Reward: {episode_reward}, Done: {done}, Win: {win}")
                return True, self.robot.history_pose
            episode_reward += r
            s = s_

        print(f"Step: {step}, Reward: {episode_reward}, Done: {done}, Win: {win}")
        return True, self.robot.history_pose
        # return False, None

    def run(self):
        """
        Running both plannig and animation.
        """
        _, history_pose = self.plan()
        print(f"path length: {len(history_pose)}")
        if not history_pose:
            raise ValueError("Path not found and planning failed!")

        path = np.array(history_pose)[:, 0:2]
        cost = np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1, keepdims=True)))
        # self.plot.plotPath(path, path_color="r", path_style="--")
        self.plot.animation(path, str(self), cost, history_pose=history_pose)

    def select_action(self, s):
        s = torch.unsqueeze(s.clone().detach(), 0)
        a = self.actor(s).detach().flatten()
        # a = self.actor(s).data.cpu().numpy().flatten()
        return a

    def optimize_model(self):
        # batch_s, batch_a, batch_r, batch_s_, batch_dw, indices, weights = self.replay_buffer.sample(self.batch_size)  # Sample a batch
        batch_s, batch_a, batch_r, batch_s_, batch_dw = self.replay_buffer.sample(self.batch_size)  # Sample a batch

        # # Normalize weights if needed, typically when alpha is not 0
        # weights = weights.unsqueeze(1)  # Reshape to match dimensions for multiplication

        # Compute the target q
        with torch.no_grad():  # target_q has no gradient
            q_ = self.critic_target(batch_s_, self.actor_target(batch_s_))
            target_q = batch_r + self.gamma * torch.logical_not(batch_dw) * q_

        # Compute the current q and the critic loss
        current_q = self.critic(batch_s, batch_a)
        critic_loss = self.criterion(target_q, current_q)
        # critic_loss = (weights * self.criterion(target_q, current_q)).mean()  # Apply weights and take mean
        # new_priorities = abs(target_q.squeeze(1) - current_q.squeeze(1)).detach() + 1e-6  # Add a small constant to avoid zero priorities
        # self.replay_buffer.update_priorities(indices, new_priorities)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute the actor loss
        actor_loss = -self.critic(batch_s, self.actor(batch_s)).mean()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # # Update priorities in the replay buffer based on the critic's TD error
        # td_error = target_q - current_q
        # new_priorities = abs(td_error.squeeze(1)).detach().cpu().numpy() + 1e-6  # Add a small constant to avoid zero priorities
        # self.replay_buffer.update_priorities(indices, new_priorities)  # Assuming you have implemented this method in ReplayBuffer

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def evaluate_policy(self):
        print(f"Evaluating: ")
        evaluate_reward = 0
        for _ in tqdm(range(self.evaluate_times)):
            s = self.reset(random_sg=True)
            done = False
            episode_reward = 0
            step = 0
            while not done:
                a = self.select_action(s)  # We do not add noise when evaluating
                s_, r, done, win = self.step(s, a)
                episode_reward += r
                s = s_
                step += 1
                if step >= self.params["MAX_ITERATION"]:
                    break
            evaluate_reward += episode_reward / step

        return evaluate_reward / self.evaluate_times

    def train(self, num_episodes=100):
        noise_std = 0.1 * torch.tensor([
            self.params["MAX_V_INC"] - self.params["MIN_V_INC"],
            self.params["MAX_W_INC"] - self.params["MIN_W_INC"]
        ], device=self.device)  # the std of Gaussian noise for exploration

        best_reward = -float('inf')

        # Train the model
        for episode in range(1, num_episodes+1):
            print(f"Episode: {episode}/{num_episodes}, Training: ")
            s = self.reset(random_sg=True)
            for episode_steps in tqdm(range(1, self.params["MAX_ITERATION"]+1)):
                if episode <= self.random_episodes:  # Take the random actions in the beginning for the better exploration
                    a = torch.tensor([
                        random.uniform(self.params["MIN_V_INC"], self.params["MAX_V_INC"]),
                        random.uniform(self.params["MIN_W_INC"], self.params["MAX_W_INC"])
                    ], device=self.device)
                else:
                    # Add Gaussian noise to actions for exploration
                    a = self.select_action(s)
                    a[0] = ((a[0] + torch.normal(0., noise_std[0].item(), size=(1,), device=self.device)).
                            clamp(self.params["MIN_V_INC"], self.params["MAX_V_INC"]))
                    a[1] = ((a[1] + torch.normal(0., noise_std[1].item(), size=(1,), device=self.device)).
                            clamp(self.params["MIN_W_INC"], self.params["MAX_W_INC"]))
                s_, r, done, win = self.step(s, a)

                self.replay_buffer.store(s, a, r, s_, win)  # Store the transition

                if win:
                    print(f"Goal reached! State: {s}, Action: {a}, Reward: {r:.4f}, Next State: {s_}")
                    break
                elif done:  # lose
                    print(f"Collision! State: {s}, Action: {a}, Reward: {r:.4f}, Next State: {s_}")
                    break

                s = s_  # Move to the next state

                # update the networks if enough samples are available
                if episode > self.random_episodes:
                    for _ in range(self.update_freq):
                        self.optimize_model()

            if episode % self.evaluate_freq == 0:
                print()
                evaluate_reward = self.evaluate_policy()
                print("Evaluate_reward:{}".format(evaluate_reward))
                print()
                self.writer.add_scalar('episode_rewards', evaluate_reward, global_step=episode)

                # Save the model
                if evaluate_reward > best_reward:
                    best_reward = evaluate_reward

                    # Create the directory if it does not exist
                    if not os.path.exists(os.path.dirname(self.actor_save_path)):
                        os.makedirs(os.path.dirname(self.actor_save_path))
                    if not os.path.exists(os.path.dirname(self.critic_save_path)):
                        os.makedirs(os.path.dirname(self.critic_save_path))

                    torch.save(self.actor.state_dict(), self.actor_save_path)
                    torch.save(self.critic.state_dict(), self.critic_save_path)

    def reset(self, random_sg=False):
        if random_sg:   # random start and goal
            start = (random.uniform(0, self.env.x_range), random.uniform(0, self.env.y_range), random.uniform(-math.pi, math.pi))
            goal = (random.uniform(0, self.env.x_range), random.uniform(0, self.env.y_range), random.uniform(-math.pi, math.pi))

            # generate random start and goal until they are not in collision
            while self.is_collision(torch.tensor(start)):
                start = (random.uniform(0, self.env.x_range), random.uniform(0, self.env.y_range), random.uniform(-math.pi, math.pi))
            while self.is_collision(torch.tensor(goal)):
                goal = (random.uniform(0, self.env.x_range), random.uniform(0, self.env.y_range), random.uniform(-math.pi, math.pi))

        else:
            start = self.start
            goal = self.goal

        self.robot = Robot(start[0], start[1], start[2], 0, 0)
        state = self.robot.state    # np.array([[self.px], [self.py], [self.theta], [self.v], [self.w]])
        state = np.pad(state, pad_width=((0, 4), (0, 0)), mode='constant')
        state[5:8, 0] = goal
        # state[8] = 0
        state = torch.tensor(state, device=self.device, dtype=torch.float).squeeze(dim=1)
        # self.total_reward = 0
        return state

    def step(self, state: torch.Tensor, action: torch.Tensor):
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
        v_d = (state[3] + action[0]).item()
        w_d = (state[4] + action[1]).item()
        self.robot.kinematic(np.array([[v_d], [w_d]]), self.params["TIME_STEP"])
        next_state = self.robot.state
        next_state = np.pad(next_state, pad_width=((0, 4), (0, 0)), mode='constant')
        next_state = torch.tensor(next_state, device=self.device, dtype=torch.float).squeeze(dim=1)
        next_state[5:8] = state[5:8]
        next_state[8] = state[8] + 1
        next_state[2] = self.regularizeAngle(next_state[2].item())
        next_state[3] = MathHelper.clamp(next_state[3].item(), self.params["MIN_V"], self.params["MAX_V"])
        next_state[4] = MathHelper.clamp(next_state[4].item(), self.params["MIN_W"], self.params["MAX_W"])
        win = self.reach_goal(tuple(next_state[0:3]), tuple(next_state[5:8]))
        lose = self.is_collision(next_state)
        reward = self.reward(next_state)
        if win:
            reward += 5.0   # self.params["MAX_ITERATION"]
        if lose:
            reward -= 1.0   # self.params["MAX_ITERATION"]
        # self.total_reward += reward
        done = win or lose
        return next_state, reward, done, win

    def is_collision(self, state):
        obstacles = np.array(list(self.obstacles))
        state = state.cpu().numpy()
        cur_pos = np.array([[state[0], state[1]]])
        obstacle_dist = np.min(cdist(obstacles, cur_pos))
        return obstacle_dist <= 0.5

    def reward(self, state):
        dist_scale = self.env.x_range + self.env.y_range
        goal_dist = self.dist((state[0], state[1]), (state[5], state[6]))
        scaled_goal_dist = goal_dist / dist_scale
        # goal_reward = 1.0 if done else 0.0

        return - 5.0 * scaled_goal_dist - state[8] / self.params["MAX_ITERATION"]

        # return goal_reward + obstacle_reward
        # return 1.0 / goal_dist - 1.0 / obstacle_dist + 1.0 * done