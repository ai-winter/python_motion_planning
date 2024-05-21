"""
@file: ddpg.py
@breif: Deep Deterministic Policy Gradient (DDPG) motion planning.
@author: Wu Maojia
@update: 2024.5.21
"""
import numpy as np
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

from .local_planner import LocalPlanner
from python_motion_planning.utils import Env, MathHelper, Robot


class ReplayBuffer(object):
    """
    Experience replay buffer to store the transitions.

    Parameters:
        state_dim (int): state dimension
        action_dim (int): action dimension
        max_size (int): maximum replay buffer size
        device (torch.device): device to store the data
    """
    def __init__(self, state_dim: int, action_dim: int, max_size: int, device: torch.device) -> None:
        self.max_size = max_size
        self.count = 0
        self.size = 0
        self.s = torch.zeros((self.max_size, state_dim), dtype=torch.float, device=device)
        self.a = torch.zeros((self.max_size, action_dim), dtype=torch.float, device=device)
        self.r = torch.zeros((self.max_size, 1), dtype=torch.float, device=device)
        self.s_ = torch.zeros((self.max_size, state_dim), dtype=torch.float, device=device)
        self.win = torch.zeros((self.max_size, 1), dtype=torch.bool, device=device)

    def store(self, s: torch.Tensor, a: torch.Tensor, r: torch.Tensor, s_: torch.Tensor, win: bool) -> None:
        """
        Store a new transition in the replay buffer.

        Parameters:
            s (torch.Tensor): state
            a (torch.Tensor): action
            r (torch.Tensor): reward
            s_ (torch.Tensor): next state
            win (bool): win or otherwise, True: win (reached the goal), False: otherwise.
        """
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.win[self.count] = torch.tensor(win, dtype=torch.bool)
        self.count = (self.count + 1) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(self.size + 1, self.max_size)  # Record the number of  transitions

    def sample(self, batch_size: int) -> tuple:
        """
        Sample a batch of transitions from the replay buffer.

        Parameters:
            batch_size (int): batch size

        Returns:
            batch_s (torch.Tensor): batch of states
            batch_a (torch.Tensor): batch of actions
            batch_r (torch.Tensor): batch of rewards
            batch_s_ (torch.Tensor): batch of next states
            batch_win (torch.Tensor): batch of win or otherwise, True: win (reached the goal), False: otherwise.
        """
        index = torch.randint(self.size, size=(batch_size,))  # Randomly sampling
        batch_s = self.s[index]
        batch_a = self.a[index]
        batch_r = self.r[index]
        batch_s_ = self.s_[index]
        batch_win = self.win[index]

        return batch_s, batch_a, batch_r, batch_s_, batch_win


class Actor(nn.Module):
    """
    Actor network to generate the action.

    Parameters:
        state_dim (int): state dimension
        action_dim (int): action dimension
        hidden_depth (int): the number of hidden layers of the neural network
        hidden_width (int): the number of neurons in hidden layers of the neural network
        min_state (torch.Tensor): minimum of each value in the state
        max_state (torch.Tensor): maximum of each value in the state
        min_action (torch.Tensor): minimum of each value in the action
        max_action (torch.Tensor): maximum of each value in the action
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_depth: int, hidden_width: int,
                 min_state: torch.Tensor, max_state: torch.Tensor, min_action: torch.Tensor, max_action: torch.Tensor) -> None:
        super(Actor, self).__init__()
        self.min_state = min_state
        self.max_state = max_state
        self.min_action = min_action
        self.max_action = max_action

        self.hidden_depth = hidden_depth
        self.input_layer = nn.Linear(state_dim, hidden_width)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_width, hidden_width) for _ in range(self.hidden_depth)])
        self.output_layer = nn.Linear(hidden_width, action_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Generate the action based on the state.

        Parameters:
            s (torch.Tensor): state

        Returns:
            a (torch.Tensor): action
        """
        # normalization
        s = (s - self.min_state) / (self.max_state - self.min_state)

        s = F.relu(self.input_layer(s))
        for i in range(self.hidden_depth):
            s = F.relu(self.hidden_layers[i](s))
        s = self.output_layer(s)
        a = self.min_action + (self.max_action - self.min_action) * torch.sigmoid(s)  # [min,max]
        return a


class Critic(nn.Module):
    """
    Critic network to estimate the value function q(s,a).

    Parameters:
        state_dim (int): state dimension
        action_dim (int): action dimension
        hidden_depth (int): the number of hidden layers of the neural network
        hidden_width (int): the number of neurons in hidden layers of the neural network
        min_state (torch.Tensor): minimum of each value in the state
        max_state (torch.Tensor): maximum of each value in the state
        min_action (torch.Tensor): minimum of each value in the action
        max_action (torch.Tensor): maximum of each value in the action
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_depth: int, hidden_width: int,
                 min_state: torch.Tensor, max_state: torch.Tensor, min_action: torch.Tensor, max_action: torch.Tensor) -> None:
        super(Critic, self).__init__()
        self.min_state = min_state
        self.max_state = max_state
        self.min_action = min_action
        self.max_action = max_action

        self.hidden_depth = hidden_depth
        self.input_layer = nn.Linear(state_dim, hidden_width)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_width, hidden_width) for _ in range(self.hidden_depth)])
        self.output_layer = nn.Linear(hidden_width, action_dim)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Q-value of (s,a)

        Parameters:
            s (torch.Tensor): state
            a (torch.Tensor): action

        Returns:
            q (torch.Tensor): Q-value of (s,a)
        """
        # normalization
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
        >>> start = (5, 5, 0)
        >>> goal = (45, 25, 0)
        >>> env = Grid(51, 31)
        >>> planner = DDPG(start, goal, env)
        >>> planner.run()

    References:
        [1] Continuous control with deep reinforcement learning
    """
    def __init__(self, start: tuple, goal: tuple, env: Env, heuristic_type: str = "euclidean",
                 hidden_depth: int = 3, hidden_width: int = 512, batch_size: int = 2e4, buffer_size: int = 2e6,
                 gamma: float = 0.999, tau: float = 1e-3, lr: float = 5e-4, train_noise: float = 0.1,
                 random_episodes: int = 100, max_episode_steps: int = 200,
                 update_freq: int = 1, update_steps: int = 5, evaluate_freq: int = 50, evaluate_episodes: int = 50,
                 actor_save_path: str = "models/actor_best.pth",
                 critic_save_path: str = "models/critic_best.pth",
                 actor_load_path: str = None,
                 critic_load_path: str = None,
                 **params) -> None:
        super().__init__(start, goal, env, heuristic_type, **params)
        # DDPG parameters
        self.hidden_depth = hidden_depth        # The number of hidden layers of the neural network
        self.hidden_width = hidden_width        # The number of neurons in hidden layers of the neural network
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
        self.actor_save_path = actor_save_path      # Save path of the trained actor network
        self.critic_save_path = critic_save_path    # Save path of the trained critic network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.n_observations = 8     # x, y, theta, v, w, g_x, g_y, g_theta
        self.n_actions = 2          # v_inc, w_inc

        self.min_state = torch.tensor([0, 0, -math.pi, self.params["MIN_V"], self.params["MIN_W"], 0, 0, -math.pi],
                                      device=self.device)
        self.max_state = torch.tensor([self.env.x_range, self.env.y_range, math.pi, self.params["MAX_V"],
                                       self.params["MAX_W"], self.env.x_range, self.env.y_range, math.pi,], device=self.device)
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

        self.actor_scheduler = ReduceLROnPlateau(self.actor_optimizer, mode='max', factor=0.2, patience=10)
        self.critic_scheduler = ReduceLROnPlateau(self.critic_optimizer, mode='max', factor=0.2, patience=10)

        self.criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(self.n_observations, self.n_actions, max_size=self.buffer_size, device=self.device)

        # Build a tensorboard
        self.writer = SummaryWriter(log_dir='runs/DDPG_{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

        # global planner
        g_start = (start[0], start[1])
        g_goal  = (goal[0], goal[1])
        self.g_planner = {"planner_name": "a_star", "start": g_start, "goal": g_goal, "env": env}
        self.path = self.g_path[::-1]
        self.history_lookahead = []

    def __del__(self) -> None:
        self.writer.close()

    def __str__(self) -> str:
        return "Deep Deterministic Policy Gradient (DDPG)"

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
            self.history_lookahead.append((lookahead_pt[0], lookahead_pt[1], theta_trj))
            s[5:7] = torch.tensor(lookahead_pt, device=self.device)
            s[7] = torch.tensor(theta_trj, device=self.device)

            a = self.select_action(s)   # get the action from the actor network
            s_, r, done, win = self.step(s, a)  # take the action and get the next state and reward
            s = s_  # Move to the next state
            self.robot.px, self.robot.py, self.robot.theta, self.robot.v, self.robot.w = tuple(s[0:5].cpu().numpy())

        return True, self.robot.history_pose

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
        self.plot.animation(path, str(self), cost, history_pose=history_pose, lookahead_pts=self.history_lookahead)

    def select_action(self, s: torch.Tensor) -> torch.Tensor:
        """
        Select the action from the actor network.

        Parameters:
            s (torch.Tensor): current state

        Returns:
            a (torch.Tensor): selected action
        """
        s = torch.unsqueeze(s.clone().detach(), 0)
        a = self.actor(s).detach().flatten()
        return a

    def optimize_model(self) -> tuple:
        """
        Optimize the neural networks when training.

        Returns:
            actor_loss (float): actor loss
            critic_loss (float): critic loss
        """
        batch_s, batch_a, batch_r, batch_s_, batch_win = self.replay_buffer.sample(self.batch_size)  # Sample a batch

        # Compute the target q
        with torch.no_grad():  # target_q has no gradient
            q_ = self.critic_target(batch_s_, self.actor_target(batch_s_))
            target_q = batch_r + self.gamma * torch.logical_not(batch_win) * q_

        # Compute the current q and the critic loss
        current_q = self.critic(batch_s, batch_a)
        critic_loss = self.criterion(target_q, current_q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0, norm_type=2)  # clip the gradient
        self.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute the actor loss
        actor_loss = -self.critic(batch_s, self.actor(batch_s)).mean()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0, norm_type=2)  # clip the gradient
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item(), critic_loss.item()

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
                a = self.select_action(s)  # We do not add noise when evaluating
                s_, r, done, win = self.step(s, a)
                self.replay_buffer.store(s, a, r, s_, win)  # Store the transition
                episode_reward += r
                s = s_
                step += 1
                if step >= self.max_episode_steps:
                    break
            evaluate_reward += episode_reward / step

        return evaluate_reward / self.evaluate_freq

    def train(self, num_episodes: int = 1000) -> None:
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
            episode_actor_loss = 0
            episode_critic_loss = 0
            for episode_steps in tqdm(range(1, self.max_episode_steps+1)):
                if episode <= self.random_episodes:
                    # Take the random actions in the beginning for the better exploration
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
                if episode > self.random_episodes and (episode_steps - 1) % self.update_steps == 0:
                    for _ in range(self.update_freq):
                        actor_loss, critic_loss = self.optimize_model()
                        episode_actor_loss += actor_loss
                        episode_critic_loss += critic_loss

            if episode > self.random_episodes:
                average_actor_loss = episode_actor_loss / (self.max_episode_steps + self.update_freq)
                average_critic_loss = episode_critic_loss / (self.max_episode_steps + self.update_freq)
                self.writer.add_scalar('Actor train loss', average_actor_loss, global_step=episode)
                self.writer.add_scalar('Critic train loss', average_critic_loss, global_step=episode)

            if episode % self.evaluate_episodes == 0:
                print()
                evaluate_reward = self.evaluate_policy()
                print("Evaluate_reward:{}".format(evaluate_reward))
                print()
                self.writer.add_scalar('Evaluate reward', evaluate_reward, global_step=episode)
                self.writer.add_scalar('Learning rate', self.actor_scheduler.optimizer.param_groups[0]['lr'],
                                       global_step=episode)     # Learning rates of the actor and critic are the same

                self.actor_scheduler.step(evaluate_reward)
                self.critic_scheduler.step(evaluate_reward)

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

    def step(self, state: torch.Tensor, action: torch.Tensor) -> tuple:
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
        v_d = (state[3] + action[0] * dt).item()
        w_d = (state[4] + action[1] * dt).item()
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
        reward = self.reward(next_state, win, lose)
        done = win or lose
        return next_state, reward, done, win

    def reward(self, state: torch.Tensor, win: bool, lose: bool) -> float:
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

        if win:
            reward += self.max_episode_steps
        if lose:
            reward -= self.max_episode_steps / 5.0

        return reward