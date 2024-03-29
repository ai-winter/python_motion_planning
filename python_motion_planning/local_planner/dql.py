"""
@file: dql.py
@breif: Deep Q-Learning motion planning
@author: Wu Maojia
@update: 2024.3.29
"""
# import osqp
import numpy as np
# from scipy import sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import math

from .local_planner import LocalPlanner
from python_motion_planning.utils import Env


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQL(LocalPlanner):
    """
    Class for Deep Q-Learning motion planning.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Env): environment
        heuristic_type (str): heuristic function type

    Examples:
        >>> from python_motion_planning.utils import Grid
        >>> from python_motion_planning.local_planner import DQL
        >>> start = (5, 5, 0)
        >>> goal = (45, 25, 0)
        >>> env = Grid(51, 31)
        >>> planner = DQL(start, goal, env)
        >>> planner.run()

    References:
        [1] Playing Atari with Deep Reinforcement Learning
        [2] Human-level control through deep reinforcement learning
    """
    def __init__(self, start: tuple, goal: tuple, env: Env, heuristic_type: str = "euclidean") -> None:
        super().__init__(start, goal, env, heuristic_type, experience_replay=True, target_network=True)
        # Deep Q-Learning parameters
        self.batch_size = 128
        self.gamma = 0.999
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000
        self.tau = 1e-3
        self.lr = 5e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_actions = 2          # v_inc, w_inc
        self.n_observations = 5     # x, y, theta, v, w

        self.policy_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayMemory(10000)

        self.steps_done = 0

        self.p = 12
        self.m = 8
        self.Q = np.diag([0.8, 0.8, 0.5])
        self.R = np.diag([2, 2])
        self.u_min = np.array([[self.params["MIN_V"]], [self.params["MIN_W"]]])
        self.u_max = np.array([[self.params["MAX_V"]], [self.params["MAX_W"]]])
        self.du_min = np.array([[self.params["MIN_V"]], [self.params["MIN_W"]]])
        self.du_max = np.array([[self.params["MAX_V_INC"]], [self.params["MAX_W_INC"]]])

        # global planner
        g_start = (start[0], start[1])
        g_goal  = (goal[0], goal[1])
        self.g_planner = {"planner_name": "a_star", "start": g_start, "goal": g_goal, "env": env}
        self.path = self.g_path[::-1]
    
    def __str__(self) -> str:
        return "Deep Q-Learning (DQL)"

    def plan(self):
        """
        Deep Q-Learning motion plan function.

        Returns:
            flag (bool): planning successful if true else failed
            pose_list (list): history poses of robot
        """
        dt = self.params["TIME_STEP"]
        u_p = (0, 0)
        for _ in range(self.params["MAX_ITERATION"]):
            # break until goal reached
            if self.shouldRotateToGoal(self.robot.position, self.goal):
                return True, self.robot.history_pose

            # get the particular point on the path at the lookahead distance
            lookahead_pt, theta_trj, kappa = self.getLookaheadPoint()

            # calculate velocity command
            e_theta = self.regularizeAngle(self.robot.theta - self.goal[2]) / 10
            if self.shouldRotateToGoal(self.robot.position, self.goal):
                if not self.shouldRotateToPath(abs(e_theta)):
                    u = np.array([[0], [0]])
                else:
                    u = np.array([[0], [self.angularRegularization(e_theta / dt)]])
            else:
                e_theta = self.regularizeAngle(
                    self.angle(self.robot.position, lookahead_pt) - self.robot.theta
                )
                if self.shouldRotateToPath(abs(e_theta), np.pi / 4):
                    u = np.array([[0], [self.angularRegularization(e_theta / dt / 10)]])
                else:
                    s = (self.robot.px, self.robot.py, self.robot.theta) # current state
                    s_d = (lookahead_pt[0], lookahead_pt[1], theta_trj)  # desired state
                    u_r = (self.robot.v, self.robot.v * kappa)           # refered input
                    u, u_p = self.dqnControl(s, s_d, u_r, u_p)

            # feed into robotic kinematic
            self.robot.kinematic(u, dt)
        
        return False, None

    def run(self):
        """
        Running both plannig and animation.
        """
        _, history_pose = self.plan()
        if not history_pose:
            raise ValueError("Path not found and planning failed!")

        path = np.array(history_pose)[:, 0:2]
        cost = np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1, keepdims=True)))
        self.plot.plotPath(self.path, path_color="r", path_style="--")
        self.plot.animation(path, str(self), cost, history_pose=history_pose)

    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        """
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(torch.FloatTensor(state).to(self.device)).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        """
        Optimize the model using experience replay.
        """
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation).
        # This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self, num_episodes=1000):
        for i_episode in range(num_episodes):
            self.robot.reset()
            state = self.robot.state
            for t in range(1000):
                action = self.select_action(state)


    def step(self, state, action):
        """
        Take a step in the environment.

        Parameters:
            state (tuple): current state of the robot
            action (int): action to take

        Returns:
            next_state (tuple): next state of the robot
            reward (float): reward for taking the action
            done (bool): whether the episode is done
        """
        v_inc, w_inc = action
        v = state[3]
        w = state[4]
        v_next = max(min(v + v_inc, self.params["MAX_V"]), self.params["MIN_V"])
        w_next = max(min(w + w_inc, self.params["MAX_W"]), self.params["MIN_W"])
        x_next, y_next, theta_next = self.robot.kinematic((v_next, w_next), self.params["TIME_STEP"])
        next_state = (x_next, y_next, theta_next, v_next, w_next)
        reward = self.reward(next_state)
        done = self.is_done(next_state)
        return next_state, reward, done

    def reward(self, state):
        return math.hypot(state[0] - self.goal[0], state[1] - self.goal[1])

    def is_done(self, state):
        e_theta = self.regularizeAngle(state[2] - self.goal[2]) / 10
        return (self.shouldRotateToGoal((state[0], state[1]), self.goal) and
                self.shouldRotateToPath(self.angle(state, self.goal), np.pi / 4))

    def dql_control(self, s: tuple, s_d: tuple, u_r: tuple, u_p: tuple) -> np.ndarray:
        pass