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
from scipy.spatial.distance import cdist
from torch.utils.tensorboard import SummaryWriter

from .local_planner import LocalPlanner
from python_motion_planning.utils import Env, MathHelper, Robot


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim):
        self.max_size = int(1e6)
        self.count = 0
        self.size = 0
        self.s = np.zeros((self.max_size, state_dim))   # state
        self.a = np.zeros((self.max_size, action_dim))  # action
        self.r = np.zeros((self.max_size, 1))           # reward
        self.s_ = np.zeros((self.max_size, state_dim))  # next state
        self.dw = np.zeros((self.max_size, 1))          # dead or win, True: win (done), False: dead (not done).

    def store(self, s, a, r, s_, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.count = (self.count + 1) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(self.size + 1, self.max_size)  # Record the number of  transitions

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
        batch_s = torch.tensor(self.s[index], dtype=torch.float)
        batch_a = torch.tensor(self.a[index], dtype=torch.float)
        batch_r = torch.tensor(self.r[index], dtype=torch.float)
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float)
        batch_dw = torch.tensor(self.dw[index], dtype=torch.float)

        return batch_s, batch_a, batch_r, batch_s_, batch_dw


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width, min_state, max_state, min_action, max_action):
        super(Actor, self).__init__()
        self.min_state = min_state
        self.max_state = max_state
        self.min_action = min_action
        self.max_action = max_action
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, action_dim)

    def forward(self, s):
        # 归一化
        s = (s - self.min_state) / (self.max_state - self.min_state)

        s = F.relu(self.l1(s))
        s = F.relu(self.l2(s))
        a = self.min_action + (self.max_action - self.min_action) * torch.sigmoid(self.l3(s))  # [min,max]
        # a = self.max_action * torch.tanh(self.l3(s))  # [-max,max]
        return a


class Critic(nn.Module):  # According to (s,a), directly calculate q(s,a)
    def __init__(self, state_dim, action_dim, hidden_width, min_state, max_state, min_action, max_action):
        super(Critic, self).__init__()
        self.min_state = min_state
        self.max_state = max_state
        self.min_action = min_action
        self.max_action = max_action
        self.l1 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, 1)

    def forward(self, s, a):
        # 归一化
        s = (s - self.min_state) / (self.max_state - self.min_state)
        a = (a - self.min_action) / (self.max_action - self.min_action)

        q = F.relu(self.l1(torch.cat([s, a], 1)))
        q = F.relu(self.l2(q))
        q = self.l3(q)
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
                         actor_save_path="models/actor_best.pth",
                         critic_save_path="models/critic_best.pth",
                         actor_load_path=None,
                         critic_load_path=None,**params) -> None:
        super().__init__(start, goal, env, heuristic_type)
        # DDPG parameters
        self.hidden_width = 256 # The number of neurons in hidden layers of the neural network
        self.batch_size = 128   # batch size
        self.gamma = 0.999      # discount factor
        self.tau = 1e-3         # Softly update the target network
        self.lr = 5e-4          # learning rate
        self.random_episodes = 5    # Take the random actions in the beginning for the better exploration
        self.update_freq = 1    # Update the network every 'update_freq' steps if episode > exploration_episodes
        self.evaluate_times = 3 # Times of evaluations and calculate the average
        self.win_reward = 1.0   # Reward for winning the game (reach the goal)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # TODO
        self.actor_save_path = actor_save_path
        self.critic_save_path = critic_save_path

        self.n_observations = 5     # x, y, theta, v, w
        self.n_actions = 2          # v_inc, w_inc

        self.min_state = torch.tensor([0, 0, -math.pi, self.params["MIN_V"], self.params["MIN_W"]])
        self.max_state = torch.tensor([self.env.x_range, self.env.y_range, math.pi, self.params["MAX_V"], self.params["MAX_W"]])
        self.min_action = torch.tensor([self.params["MIN_V_INC"], self.params["MIN_W_INC"]])
        self.max_action = torch.tensor([self.params["MAX_V_INC"], self.params["MAX_W_INC"]])

        self.actor = Actor(self.n_observations, self.n_actions, self.hidden_width, self.min_state, self.max_state,
                           self.min_action, self.max_action)
        if actor_load_path:
            self.actor.load_state_dict(torch.load(actor_load_path))
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(self.n_observations, self.n_actions, self.hidden_width, self.min_state, self.max_state,
                             self.min_action, self.max_action)
        if critic_load_path:
            self.critic.load_state_dict(torch.load(critic_load_path))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(self.n_observations, self.n_actions)

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
            s_, r, done = self.step(s, a)
            if done:
                return True, self.robot.history_pose
            episode_reward += r
            s = s_

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
        a = self.actor(s).data.numpy().flatten()
        return a

    def optimize_model(self):
        batch_s, batch_a, batch_r, batch_s_, batch_dw = self.replay_buffer.sample(self.batch_size)  # Sample a batch

        # Compute the target q
        with torch.no_grad():  # target_q has no gradient
            q_ = self.critic_target(batch_s_, self.actor_target(batch_s_))
            target_q = batch_r + self.gamma * (1 - batch_dw) * q_

        # Compute the current q and the critic loss
        current_q = self.critic(batch_s, batch_a)
        critic_loss = self.criterion(target_q, current_q)
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

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def evaluate_policy(self):
        print(f"Evaluating: ")
        evaluate_reward = 0
        for _ in tqdm(range(self.evaluate_times)):
            s = self.reset()
            done = False
            episode_reward = 0
            step = 0
            while not done:
                a = self.select_action(s)  # We do not add noise when evaluating
                s_, r, done = self.step(s, a)
                episode_reward += r
                s = s_
                step += 1
                if step >= self.params["MAX_ITERATION"]:
                    break
            evaluate_reward += episode_reward / step

        return evaluate_reward / self.evaluate_times

    def train(self, num_episodes=100):
        noise_std = 0.1 * np.array([
            self.params["MAX_V_INC"] - self.params["MIN_V_INC"],
            self.params["MAX_W_INC"] - self.params["MIN_W_INC"]
        ])  # the std of Gaussian noise for exploration

        best_reward = -float('inf')

        # Train the model
        for episode in range(1, num_episodes+1):
            print(f"Episode: {episode}/{num_episodes}, Training: ")
            s = self.reset()
            episode_steps = 0
            for episode_steps in tqdm(range(1, self.params["MAX_ITERATION"]+1)):
                if episode <= self.random_episodes:  # Take the random actions in the beginning for the better exploration
                    # a = torch.tensor(self.sample_action())
                    a = torch.tensor([
                        random.uniform(self.params["MIN_V_INC"], self.params["MAX_V_INC"]),
                        random.uniform(self.params["MIN_W_INC"], self.params["MAX_W_INC"])
                    ])
                else:
                    # Add Gaussian noise to actions for exploration
                    a = self.select_action(s)
                    a[0] = (a[0] + np.random.normal(0, noise_std[0])).clip(self.params["MIN_V_INC"], self.params["MAX_V_INC"])
                    a[1] = (a[1] + np.random.normal(0, noise_std[1])).clip(self.params["MIN_W_INC"], self.params["MAX_W_INC"])
                s_, r, done = self.step(s, a)

                self.replay_buffer.store(s, a, r, s_, done)  # Store the transition

                # if episode_steps <= 5 or self.params["MAX_ITERATION"] - episode_steps < 5:
                #     print(f"Step: {episode_steps}, State: {s}, Action: {a}, Reward: {r:.4f}, Next State: {s_}")
                if done:
                    print(f"Goal reached! State: {s}, Action: {a}, Reward: {r:.4f}, Next State: {s_}")
                    break

                s = s_  # Move to the next state

                # update the networks if enough samples are available
                if episode > self.random_episodes:
                    for _ in range(self.update_freq):
                        self.optimize_model()

            evaluate_reward = self.evaluate_policy()
            print("episode_steps:{} \t evaluate_reward:{}".format(episode_steps, evaluate_reward))
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

    def reset(self):
        self.robot = Robot(self.start[0], self.start[1], self.start[2], 0, 0)
        state = torch.tensor(self.robot.state, device=self.device, dtype=torch.float).squeeze(dim=1)
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
        next_state = torch.tensor(self.robot.state, device=self.device, dtype=torch.float).squeeze(dim=1)
        next_state[2] = self.regularizeAngle(next_state[2].item())
        next_state[3] = MathHelper.clamp(next_state[3].item(), self.params["MIN_V"], self.params["MAX_V"])
        next_state[4] = MathHelper.clamp(next_state[4].item(), self.params["MIN_W"], self.params["MAX_W"])
        done = self.reach_goal(tuple(next_state[0:3]), tuple(self.goal[0:3]))
        reward = self.reward(next_state, done)
        return next_state, reward, done

    def reward(self, state, done):
        dist_scale = self.env.x_range + self.env.y_range

        scaled_goal_dist = self.dist((state[0], state[1]), (self.goal[0], self.goal[1])) / dist_scale

        obstacles = np.array(list(self.obstacles))
        cur_pos = np.array([[state[0], state[1]]])
        obstacle_dist = np.min(cdist(obstacles, cur_pos))

        return - scaled_goal_dist - 1.0 / obstacle_dist + 1.0 * done