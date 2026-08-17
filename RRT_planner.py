"""
RRT 2D 动态取样路径规划器 - 用于ARVP围捕任务的引导
核心特性：
  1. 带移动障碍物碰撞检测的RRT规划
  2. 围捕锚点分配（三机120°均匀分布在目标周围）
  3. 路径短截取前瞻参考点（动态取样）
"""
import numpy as np
import random
from math import cos, sin, atan2, sqrt, pi


class RRTNode:
    __slots__ = ('x', 'y', 'parent')

    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent


class RRTPlanner2D:
    def __init__(self, bounds, max_iter=500, step_size=0.06,
                 goal_sample_rate=0.25, robot_radius=0.1):
        """
        :param bounds: 场景边界 [x_min, x_max, y_min, y_max]
        :param max_iter: 单次规划最大迭代
        :param step_size: RRT扩展步长 (m)
        :param goal_sample_rate: 以目标点为采样点的概率
        :param robot_radius: 机器人安全半径
        """
        self.bounds = bounds
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.robot_radius = robot_radius

    def plan(self, start, goal, obstacles, obstacle_radii):
        """
        RRT规划主函数
        :param start: [x, y] 起点
        :param goal:  [x, y] 终点（围捕锚点）
        :param obstacles: list[[x,y],...] 障碍物中心位置
        :param obstacle_radii: list[r,...] 障碍物半径
        :return: 路径 list[[x,y],...]；规划失败返回None
        """
        start_node = RRTNode(start[0], start[1])
        goal_node = RRTNode(goal[0], goal[1])
        tree = [start_node]

        for _ in range(self.max_iter):
            # Sampling
            if random.random() < self.goal_sample_rate:
                rnd = [goal_node.x, goal_node.y]
            else:
                rnd = [
                    random.uniform(self.bounds[0], self.bounds[1]),
                    random.uniform(self.bounds[2], self.bounds[3])
                ]

            # Nearest
            nearest_idx = self._nearest_index(tree, rnd)
            nearest_node = tree[nearest_idx]

            # Steer
            new_node = self._steer(nearest_node, RRTNode(rnd[0], rnd[1]))

            # Collision check
            if not self._is_collision_free(nearest_node, new_node, obstacles, obstacle_radii):
                continue

            tree.append(new_node)

            # Goal check
            if self._dist(new_node, goal_node) <= self.step_size:
                final_node = self._steer(new_node, goal_node)
                if self._is_collision_free(new_node, final_node, obstacles, obstacle_radii):
                    tree.append(final_node)
                    return self._extract_path(tree, final_node)

        return None  # failed

    # ---------- anchor allocation ----------
    @staticmethod
    def allocate_anchors(target_pos, drone_positions, capture_radius=0.36):
        """
        为三架围捕无人机分配目标圆周上的围捕锚点（严格120°一一对应）
        分配策略：按无人机相对于目标的方位角顺时针排序，
        对应地分配锚点角度，保证每个锚点唯一且环绕目标。
        :param target_pos: [x,y] 目标当前位置
        :param drone_positions: list[[x,y], ...] n架无人机当前位置
        :param capture_radius: 锚点圆周半径
        :return: list[[x,y], ...] 锚点列表（与drone_positions一一对应，无重复）
        """
        pe = np.array(target_pos)
        n = len(drone_positions)
        # 每个无人机相对于目标的方位角
        angles = []
        for p in drone_positions:
            vec = np.array(p) - pe
            angles.append(atan2(vec[1], vec[0]))
        # 按方位角从小到大排序无人机索引（顺时针序从-π到π）
        order = sorted(range(n), key=lambda k: angles[k])  # order[k] = 排在第k个方位的无人机idx
        # 以排第一个无人机的方位角为起点，按间隔 2π/n 生成锚点角度。
        # 轻微偏移：从第一个无人机角度后退 π/n，让锚点序列"居中"
        base_angle = angles[order[0]] - pi / n
        anchor_angles = [base_angle + 2 * pi * k / n for k in range(n)]
        # 归一化到 [-π, π]
        anchor_angles = [((a + pi) % (2 * pi)) - pi for a in anchor_angles]
        # 一一对应：order[k] 号无人机 ↔ anchor_angles[k]
        anchors_out = [None] * n
        for k in range(n):
            drone_idx = order[k]
            ang = anchor_angles[k]
            anchors_out[drone_idx] = [
                float(pe[0] + capture_radius * cos(ang)),
                float(pe[1] + capture_radius * sin(ang))
            ]
        return anchors_out

    # ---------- dynamic sampling: short segment extraction ----------
    @staticmethod
    def extract_reference(path, current_pos, look_ahead_dist):
        """
        从RRT路径中动态取样：取距离当前位置前瞻距离处的点作为参考点
        :param path: list[[x,y], ...] RRT路径（起点到终点）
        :param current_pos: [x,y] 当前无人机位置
        :param look_ahead_dist: 前瞻距离（建议= k*v_max*dt, k=3~5）
        :return: [x,y] 参考点；路径无效返回None
        """
        if path is None or len(path) == 0:
            return None
        # 找到路径上距离current_pos最近的点索引
        cur = np.array(current_pos)
        dists = [np.linalg.norm(np.array(p) - cur) for p in path]
        start_idx = int(np.argmin(dists))
        # 从start_idx向前累积距离，达到look_ahead_dist所在位置
        cum = 0.0
        for i in range(start_idx, len(path) - 1):
            seg = np.linalg.norm(np.array(path[i + 1]) - np.array(path[i]))
            if cum + seg >= look_ahead_dist:
                remain = look_ahead_dist - cum
                ratio = remain / seg if seg > 1e-8 else 0.0
                wp = np.array(path[i]) + ratio * (np.array(path[i + 1]) - np.array(path[i]))
                return [float(wp[0]), float(wp[1])]
            cum += seg
        # 前瞻距离超出路径长度，返回路径终点
        return list(path[-1])

    # ---------- internal helpers ----------
    @staticmethod
    def _dist(n1, n2):
        return sqrt((n1.x - n2.x) ** 2 + (n1.y - n2.y) ** 2)

    def _nearest_index(self, tree, rnd):
        dlist = [sqrt((n.x - rnd[0]) ** 2 + (n.y - rnd[1]) ** 2) for n in tree]
        return int(np.argmin(dlist))

    def _steer(self, from_node, to_node):
        d = self._dist(from_node, to_node)
        if d <= self.step_size:
            return RRTNode(to_node.x, to_node.y, parent=from_node)
        ratio = self.step_size / d
        new_x = from_node.x + (to_node.x - from_node.x) * ratio
        new_y = from_node.y + (to_node.y - from_node.y) * ratio
        return RRTNode(new_x, new_y, parent=from_node)

    def _is_collision_free(self, n1, n2, obstacles, obstacle_radii):
        """线段 n1->n2 是否与所有障碍物无碰撞"""
        # 边界检查
        if not (self.bounds[0] <= n2.x <= self.bounds[1] and
                self.bounds[2] <= n2.y <= self.bounds[3]):
            return False
        A = np.array([n1.x, n1.y])
        B = np.array([n2.x, n2.y])
        for obs, r in zip(obstacles, obstacle_radii):
            C = np.array(obs)
            R = r + self.robot_radius
            # 点C到线段AB的最短距离
            AB = B - A
            t = np.dot(C - A, AB) / (np.dot(AB, AB) + 1e-9)
            t = max(0.0, min(1.0, t))
            proj = A + t * AB
            if np.linalg.norm(C - proj) <= R:
                return False
        return True

    @staticmethod
    def _extract_path(tree, end_node):
        path = []
        node = end_node
        while node is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.reverse()
        return path
