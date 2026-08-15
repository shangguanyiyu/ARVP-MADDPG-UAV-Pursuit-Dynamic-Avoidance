import numpy as np
from typing import List, Optional, Tuple


class RRTStar2D:
    def __init__(self,
                 bounds: Tuple[float, float],
                 circular_obstacles: Optional[List[dict]] = None,
                 robot_radius: float = 0.06,
                 max_iter: int = 400,
                 step_size: float = 0.12,
                 goal_radius: float = 0.15,
                 neighbor_radius: float = 0.25,
                 goal_sample_rate: float = 0.15):
        self.bounds = bounds
        self.circular_obstacles = circular_obstacles if circular_obstacles else []
        self.robot_radius = robot_radius
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_radius = goal_radius
        self.neighbor_radius = neighbor_radius
        self.goal_sample_rate = goal_sample_rate

        self.nodes = []  # list of dict: {'pos': np.array, 'parent': int, 'cost': float}

    def _random_sample(self, goal: np.ndarray) -> np.ndarray:
        if np.random.random() < self.goal_sample_rate:
            return goal.copy()
        x = np.random.uniform(0.0, self.bounds[0])
        y = np.random.uniform(0.0, self.bounds[1])
        return np.array([x, y])

    def _nearest_node_idx(self, pos: np.ndarray) -> int:
        if not self.nodes:
            return 0
        dists = np.linalg.norm(
            np.array([n['pos'] for n in self.nodes]) - pos, axis=1
        )
        return int(np.argmin(dists))

    def _near_indices(self, pos: np.ndarray) -> List[int]:
        if not self.nodes:
            return []
        dists = np.linalg.norm(
            np.array([n['pos'] for n in self.nodes]) - pos, axis=1
        )
        return [int(i) for i, d in enumerate(dists) if d < self.neighbor_radius]

    def _steer(self, from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
        direction = to_pos - from_pos
        dist = np.linalg.norm(direction)
        if dist <= self.step_size:
            return to_pos.copy()
        unit = direction / (dist + 1e-9)
        return from_pos + unit * self.step_size

    def _is_segment_free(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        if not self._is_point_free(p1) or not self._is_point_free(p2):
            return False
        steps = int(np.ceil(np.linalg.norm(p2 - p1) / (self.step_size * 0.25)))
        if steps < 2:
            steps = 2
        for t in np.linspace(0, 1, steps):
            pt = p1 + t * (p2 - p1)
            if not self._is_point_free(pt):
                return False
        return True

    def _is_point_free(self, pos: np.ndarray) -> bool:
        x, y = pos[0], pos[1]
        r = self.robot_radius
        if x - r < 0 or x + r > self.bounds[0]:
            return False
        if y - r < 0 or y + r > self.bounds[1]:
            return False
        for obs in self.circular_obstacles:
            ox, oy = obs['position'][0], obs['position'][1]
            combined_r = obs['radius'] + r
            if (x - ox) ** 2 + (y - oy) ** 2 < combined_r ** 2:
                return False
        return True

    def plan(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        start = np.array(start, dtype=float)
        goal = np.array(goal, dtype=float)

        self.nodes = [{'pos': start.copy(), 'parent': -1, 'cost': 0.0}]

        best_goal_idx = -1
        best_goal_cost = np.inf

        for _ in range(self.max_iter):
            rand_pt = self._random_sample(goal)

            nearest_idx = self._nearest_node_idx(rand_pt)
            nearest = self.nodes[nearest_idx]
            new_pos = self._steer(nearest['pos'], rand_pt)

            if not self._is_segment_free(nearest['pos'], new_pos):
                continue

            new_cost = nearest['cost'] + np.linalg.norm(new_pos - nearest['pos'])

            near_indices = self._near_indices(new_pos)

            for ni in near_indices:
                near_node = self.nodes[ni]
                d = np.linalg.norm(new_pos - near_node['pos'])
                if (near_node['cost'] + d < new_cost
                        and self._is_segment_free(near_node['pos'], new_pos)):
                    nearest_idx = ni
                    new_cost = near_node['cost'] + d

            new_node = {'pos': new_pos, 'parent': nearest_idx, 'cost': new_cost}
            new_idx = len(self.nodes)
            self.nodes.append(new_node)

            for ni in near_indices:
                if ni == nearest_idx:
                    continue
                near_node = self.nodes[ni]
                d = np.linalg.norm(new_pos - near_node['pos'])
                if (new_cost + d < near_node['cost']
                        and self._is_segment_free(new_pos, near_node['pos'])):
                    near_node['parent'] = new_idx
                    near_node['cost'] = new_cost + d

            dist_to_goal = np.linalg.norm(new_pos - goal)
            if dist_to_goal < self.goal_radius and new_cost < best_goal_cost:
                final_segment_ok = self._is_segment_free(new_pos, goal)
                if final_segment_ok:
                    best_goal_idx = new_idx
                    best_goal_cost = new_cost + dist_to_goal

        if best_goal_idx < 0:
            goal_reached_idxs = [
                i for i, n in enumerate(self.nodes)
                if np.linalg.norm(n['pos'] - goal) < self.goal_radius * 2.0
            ]
            if goal_reached_idxs:
                costs = [self.nodes[i]['cost'] for i in goal_reached_idxs]
                best_goal_idx = goal_reached_idxs[int(np.argmin(costs))]

        if best_goal_idx < 0:
            return self._straight_line_path(start, goal)

        path_rev = [goal.copy()]
        cur = best_goal_idx
        while cur >= 0:
            path_rev.append(self.nodes[cur]['pos'])
            cur = self.nodes[cur]['parent']
        path = list(reversed(path_rev))

        if len(path) < 2:
            return self._straight_line_path(start, goal)
        return path

    def _straight_line_path(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        steps = int(np.ceil(np.linalg.norm(goal - start) / self.step_size))
        steps = max(steps, 2)
        return [start + t * (goal - start) for t in np.linspace(0, 1, steps)]


def sample_waypoints_along_path(path: List[np.ndarray],
                                current_pos: np.ndarray,
                                num_waypoints: int = 4,
                                spacing: Optional[float] = None) -> List[np.ndarray]:
    if len(path) < 2:
        straight = np.array(path[0]) if path else current_pos
        direction = np.array([1.0, 0.0])
        waypoints = []
        for i in range(1, num_waypoints + 1):
            waypoints.append(current_pos + direction * 0.1 * i)
        return waypoints

    path_arr = np.array(path)
    seg_lens = np.linalg.norm(np.diff(path_arr, axis=0), axis=1)
    cum_lens = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cum_lens[-1]

    if spacing is None:
        spacing = total_len / (num_waypoints + 1)

    start_idx = 0
    d_start = np.inf
    for i, pt in enumerate(path_arr):
        d = np.linalg.norm(pt - current_pos)
        if d < d_start:
            d_start = d
            start_idx = i
    start_s = cum_lens[min(start_idx, len(cum_lens) - 1)]

    waypoints = []
    for k in range(1, num_waypoints + 1):
        target_s = start_s + k * spacing
        if target_s >= total_len:
            waypoints.append(path_arr[-1].copy())
            continue
        seg_idx = int(np.searchsorted(cum_lens, target_s) - 1)
        seg_idx = max(0, min(seg_idx, len(path_arr) - 2))
        s0 = cum_lens[seg_idx]
        s1 = cum_lens[seg_idx + 1]
        alpha = (target_s - s0) / max(s1 - s0, 1e-9)
        pt = path_arr[seg_idx] + alpha * (path_arr[seg_idx + 1] - path_arr[seg_idx])
        waypoints.append(pt.copy())
    return waypoints


def compute_encircle_positions(target_pos: np.ndarray,
                               num_points: int = 3,
                               encircle_radius: float = 0.25) -> List[np.ndarray]:
    points = []
    for i in range(num_points):
        angle = 2.0 * np.pi * i / num_points
        x = target_pos[0] + encircle_radius * np.cos(angle)
        y = target_pos[1] + encircle_radius * np.sin(angle)
        points.append(np.array([x, y]))
    return points


def encode_waypoints_for_obs(waypoints: List[np.ndarray],
                             current_pos: np.ndarray,
                             world_size: float,
                             num_expected: int = 4) -> List[float]:
    feats = []
    norm_scale = world_size * np.sqrt(2.0) + 1e-9
    while len(waypoints) < num_expected:
        waypoints = waypoints + [waypoints[-1].copy() if waypoints else current_pos.copy()]
    total_dist = 0.0
    prev = current_pos
    for wp in waypoints:
        total_dist += np.linalg.norm(wp - prev)
        prev = wp
    running = 0.0
    prev = current_pos
    for wp in waypoints[:num_expected]:
        rel = wp - current_pos
        dx = rel[0] / norm_scale
        dy = rel[1] / norm_scale
        step_d = np.linalg.norm(wp - prev)
        running += step_d
        progress = running / max(total_dist, 1e-6)
        feats.extend([dx, dy, progress])
        prev = wp
    return feats


def encode_hunter_threat_for_target(hunter_positions: List[np.ndarray],
                                    target_pos: np.ndarray,
                                    world_size: float,
                                    rrt_paths: Optional[List[List[np.ndarray]]] = None) -> List[float]:
    feats = []
    norm_scale = world_size * np.sqrt(2.0) + 1e-9
    for i, hpos in enumerate(hunter_positions):
        if rrt_paths and i < len(rrt_paths) and rrt_paths[i] and len(rrt_paths[i]) > 1:
            first_segment = rrt_paths[i][1] - hpos
            angle = np.arctan2(first_segment[1], first_segment[0])
        else:
            direc = target_pos - hpos
            angle = np.arctan2(direc[1], direc[0])
        d = np.linalg.norm(target_pos - hpos) / norm_scale
        feats.extend([np.sin(angle), np.cos(angle), d])
    return feats
