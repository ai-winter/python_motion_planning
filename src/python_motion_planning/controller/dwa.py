from typing import List, Tuple, Optional
import numpy as np
from itertools import product
from scipy.spatial.distance import cdist

from .pure_pursuit import PurePursuit


class DWA(PurePursuit):
    """
    Dynamic Window Approach (DWA) controller.

    Notes:
        - observation `obs` is expected to be: [pos( dim ), vel( dim ), ...]
        - action returned is acceleration vector (dim,)
        - target returned is the lookahead point (from PurePursuit._get_lookahead_point)

    Parameters:
        observation_space: gymnasium observation space
        action_space: gymnasium Box for acceleration (shape=(dim,))
        path: list of path points (each is a tuple of length dim)
        dt: control timestep
        max_speed: maximum speed magnitude (optional, used by clip_velocity)
        lookahead_distance: lookahead distance for lookahead point (inherited)
        predict_time: how long to predict forward (seconds)
        heading_weight: weight for heading (distance-to-target) cost
        obstacle_weight: weight for obstacle cost (larger -> avoid obstacles more)
        velocity_weight: weight for velocity (prefer larger speed if positive)
        v_resolution: resolution when sampling velocities per-dimension
        max_samples: maximum number of candidate velocities to evaluate (to avoid combinatorial explosion)
        obstacles: optional np.ndarray of shape (M, dim) with obstacle points
    """
    def __init__(self,
                 observation_space,
                 action_space,
                 path: List[Tuple[float, ...]],
                 dt: float,
                 max_speed: float = np.inf,
                 lookahead_distance: float = 2.0,
                 predict_time: float = 1.5,
                 heading_weight: float = 1.0,
                 obstacle_weight: float = 1.0,
                 velocity_weight: float = 0.1,
                 v_resolution: float = 0.1,
                 max_samples: int = 1024,
                 obstacles: Optional[np.ndarray] = None):
        super().__init__(observation_space, action_space, path, dt, max_speed, lookahead_distance)

        self.predict_time = predict_time
        self.heading_weight = heading_weight
        self.obstacle_weight = obstacle_weight
        self.velocity_weight = velocity_weight
        self.v_resolution = v_resolution
        self.max_samples = max_samples

        # obstacles: None or array shape (M, dim)
        self.obstacles = None if obstacles is None else np.asarray(obstacles)

    def reset(self):
        super().reset()
        # nothing else to reset for now

    def set_obstacles(self, obstacles: Optional[np.ndarray]):
        """Set/replace obstacle list used by evaluation."""
        self.obstacles = None if obstacles is None else np.asarray(obstacles)

    def get_action(self, obs: np.ndarray) -> Tuple[np.ndarray, tuple]:
        """
        Overrides PurePursuit.get_action: run DWA search and return acceleration + target.

        Parameters:
            obs: observation array ([pos, vel, ...]) length at least 2*dim

        Returns:
            acc: acceleration vector (dim,)
            target: lookahead point (ndarray of length dim)
        """
        if self.goal is None:
            return np.zeros(self.action_space.shape), self.goal

        dim = self.action_space.shape[0]
        pos = obs[:dim].astype(float)
        vel = obs[dim:2*dim].astype(float)

        # get lookahead target from parent
        target = self._get_lookahead_point(pos)

        # build dynamic window in velocity space for each dimension
        # v_min = current_vel + acc_min * dt, v_max = current_vel + acc_max * dt
        acc_min = self.action_space.low
        acc_max = self.action_space.high

        # per-dim v ranges
        v_mins = vel + acc_min * self.dt
        v_maxs = vel + acc_max * self.dt

        # ensure reasonable ordering
        v_low = np.minimum(v_mins, v_maxs)
        v_high = np.maximum(v_mins, v_maxs)

        # for sampling we will generate per-dim linspace
        grids = []
        counts = []
        for i in range(dim):
            span = float(v_high[i] - v_low[i])
            if span <= 1e-8:
                # degenerate: single value
                grid = np.array([v_low[i]])
            else:
                num = max(2, int(np.ceil(span / self.v_resolution)) + 1)
                grid = np.linspace(v_low[i], v_high[i], num=num)
            grids.append(grid)
            counts.append(len(grid))

        # estimate total number of combos
        total = int(np.prod(counts))
        candidates = []

        # generate candidate velocities (avoid exploding)
        if total <= self.max_samples:
            # full grid
            for comb in product(*grids):
                candidates.append(np.array(comb, dtype=float))
        else:
            # sample randomly from hyper-rectangle uniformly (including some grid points)
            # sample half from uniform, half from grid anchors
            rng = np.random.default_rng()
            n_rand = self.max_samples
            # uniform sample within v_low..v_high
            samples = rng.random((n_rand, dim)) * (v_high - v_low) + v_low
            candidates = [s.astype(float) for s in samples]

        # Evaluate candidates
        best_cost = float("inf")
        best_vel = vel.copy()

        # Precompute prediction steps
        steps = max(1, int(np.ceil(self.predict_time / self.dt)))
        traj_dt = self.dt

        # If obstacles exist, convert to numpy
        obs_pts = None if self.obstacles is None else np.asarray(self.obstacles)

        for v_cand in candidates:
            # Clip candidate velocity magnitude to overall max_speed if defined (preserve direction)
            v_cand = self.clip_velocity(v_cand)

            # Predict trajectory assuming constant velocity v_cand
            # simple kinematic: pos_t+1 = pos_t + v_cand * dt
            traj = np.zeros((steps, dim))
            p = pos.copy()
            for s in range(steps):
                p = p + v_cand * traj_dt
                traj[s, :] = p

            final_pos = traj[-1]

            # cost: heading = distance from final_pos to lookahead target (smaller better)
            heading_cost = np.linalg.norm(final_pos - target)

            # obstacle cost: minimal distance from any obstacle to any traj point (smaller -> higher penalty)
            if obs_pts is None or obs_pts.size == 0:
                obstacle_cost = 0.0
            else:
                # compute pairwise distances (M x steps) and take min
                D = cdist(obs_pts, traj)  # shape (M, steps)
                min_D = float(np.min(D))
                # map to cost: smaller distance -> larger cost. we use inverse-like transform
                # avoid division by zero
                obstacle_cost = 0.0 if min_D >= 1e6 else 1.0 / (min_D + 1e-6)

            # velocity cost: prefer larger speed (we will negate so smaller total better)
            velocity_score = -np.linalg.norm(v_cand)

            # weighted sum (note obstacle_cost grows when close -> increases total cost)
            total_cost = (self.heading_weight * heading_cost +
                          self.obstacle_weight * obstacle_cost +
                          self.velocity_weight * velocity_score)

            if total_cost < best_cost:
                best_cost = total_cost
                best_vel = v_cand

        # compute acceleration command
        acc = (best_vel - vel) / self.dt

        # clip to action bounds and return
        acc = self.clip_action(acc)
        best_vel = self.clip_velocity(best_vel)

        return acc, tuple(target)
