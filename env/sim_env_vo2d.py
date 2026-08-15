import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.cm as cm
import matplotlib.image as mpimg
from gymnasium import spaces

from math_tool import *
import matplotlib.backends.backend_agg as agg
from PIL import Image
import random
import copy
from math import ceil, floor, sqrt
import copy
from RVO import distance, in_between
from math import cos, sin, tan, atan2, asin
import time

from math import pi as PI


class UAVEnv:
    def __init__(self, length=2, num_obstacle=4, num_agents=4):
        self.length = length
        self.num_obstacle = num_obstacle
        self.num_agents = num_agents
        self.time_step = 0.5
        self.v_max = 0.12
        self.v_max_e = 0.03
        self.a_max = 0.06
        self.a_max_e = 0.04
        self.L_sensor = 0.2
        self.num_lasers = 16
        self.multi_current_lasers = [[self.L_sensor for _ in range(self.num_lasers)] for _ in range(self.num_agents)]
        self.agents = ['agent_0', 'agent_1', 'agent_2', 'target']
        self.info = np.random.get_state()
        self.obstacles = [
            obstacle(mode='random', index=i, total_obstacles=self.num_obstacle)
            for i in range(self.num_obstacle)
        ]
        self.history_positions = [[] for _ in range(num_agents)]
        self.obs_vel = [obs.velocity for obs in self.obstacles]
        self.last_pos = [np.zeros(2) for _ in range(num_agents)]

        # ===== 方案三：VO2D 新增参数 =====
        # 奖励系数
        self.mu8 = -0.2   # VO 修正惩罚系数
        self.mu9 = 0.05   # 预测性避碰加分系数
        self.TTC_threshold = 1.0  # TTC 阈值 (秒)

        # VO 投影 & 观测辅助缓存（每 step 重置一次）
        self.vo_correction_magnitudes = [0.0] * self.num_agents
        self.vo_pred_avoid_flags = [False] * self.num_agents  # 是否满足 mu9 加分条件
        self.vo_min_ttc_each = [np.inf] * self.num_agents

        # VO 2D 观测参数
        self.vo_prox_threshold = 0.8  # 认为 VO 锥"活跃"的距离阈值

        self.action_space = {
            'agent_0': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'agent_1': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'agent_2': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'target': spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        }
        # ===== 方案三：观测维度扩充 =====
        # hunter (agt): 原 26 + VO2D特征6 = 32
        # target (tar): 原 23 + VO2D特征6 = 29
        self.observation_space = {
            'agent_0': spaces.Box(low=-np.inf, high=np.inf, shape=(32,)),
            'agent_1': spaces.Box(low=-np.inf, high=np.inf, shape=(32,)),
            'agent_2': spaces.Box(low=-np.inf, high=np.inf, shape=(32,)),
            'target': spaces.Box(low=-np.inf, high=np.inf, shape=(29,))
        }

    # =========================================================
    # 方案三（1）：VO 锥几何辅助函数 & 2D 立体锥观测编码
    # =========================================================

    def compute_vo_cone(self, pA, vA, pB, vB, combined_radius):
        """
        计算 A 相对于 B 的单个 VO (平移圆锥) 的几何参数。
        Returns:
            dict or None: {
                'center_theta': 锥中心方向 (相对 A),
                'half_angle': 锥半角 (弧度),
                'distance': 中心距离 dist_BA,
                'ttc': 预计碰撞时间,
                'vo_origin': 平移VO的顶点 (transl_vB_vA) - 即 [pA + vB],
                'bound_left_theta': 左边界方向角,
                'bound_right_theta': 右边界方向角,
                'in_vo': 当前速度 vA 是否在锥内,
                'dist_vo_edge': 若在锥内, 到最近边界的角度距离 (弧度, >=0); 否则 0
            }
            距离过近导致 asin 域溢出时自动截断。
        """
        transl_vB_vA = [pA[0] + vB[0], pA[1] + vB[1]]
        dist_BA = distance(pA, pB)

        if combined_radius > dist_BA:
            dist_BA_clamped = combined_radius
        else:
            dist_BA_clamped = dist_BA

        sin_arg = combined_radius / dist_BA_clamped
        if sin_arg > 1.0:
            sin_arg = 1.0
        if sin_arg < -1.0:
            sin_arg = -1.0

        theta_BA = atan2(pB[1] - pA[1], pB[0] - pA[0])
        theta_BAort = asin(sin_arg)

        theta_ort_left = theta_BA + theta_BAort
        theta_ort_right = theta_BA - theta_BAort

        # 计算当前速度是否在锥内 (VO: 顶点 = pA + vB)
        dif = [vA[0] + pA[0] - transl_vB_vA[0], vA[1] + pA[1] - transl_vB_vA[1]]
        theta_dif = atan2(dif[1], dif[0])
        theta_right = theta_ort_right
        theta_left = theta_ort_left

        # 锥中心方向 (相对A的坐标系, 是从A指向VO内部的方向)
        theta_center = (theta_left + theta_right) / 2
        if theta_left > theta_right:
            theta_center = (theta_left + theta_right + 2 * PI) / 2
            if theta_center > PI:
                theta_center -= 2 * PI

        in_vo = in_between(theta_right, theta_dif, theta_left)

        # 到最近边界的角度距离 (若在锥外=0)
        dist_vo_edge = 0.0
        if in_vo:
            # 计算 theta_dif 到 left/right 的最小(有向)夹角
            def ang_diff(a, b):
                d = a - b
                while d > PI:
                    d -= 2 * PI
                while d < -PI:
                    d += 2 * PI
                return abs(d)
            dL = ang_diff(theta_left, theta_dif)
            dR = ang_diff(theta_right, theta_dif)
            dist_vo_edge = min(dL, dR)

        # TTC
        ttc = self.calculate_expected_collision_time(pA, vA, pB, vB, combined_radius)

        return {
            'center_theta': theta_center,
            'half_angle': theta_BAort,
            'distance': distance(pA, pB),
            'ttc': ttc,
            'vo_origin': transl_vB_vA,
            'bound_left_theta': theta_ort_left,
            'bound_right_theta': theta_ort_right,
            'in_vo': in_vo,
            'dist_vo_edge': dist_vo_edge,
        }

    def collect_all_vo_cones(self, agent_idx, pA, vA):
        """
        为 agent_idx 收集所有(障碍物/友军/target)的VO锥列表。
        返回 (RVO_BA_all, cones_info)
            RVO_BA_all: 兼容 intersect() 格式的列表 (用于投影)
            cones_info: 每个锥的详细字典列表 (用于观测编码)
        """
        ROB_RAD = 0.1 - 0.05  # 与 VO_reward 系列一致
        OVER_APPROX_C2S = 1.3
        ws_model = self.get_ws_model()

        RVO_BA_all = []
        cones_info = []

        # ---- 友军 (其他 UAV) ----
        for j in range(self.num_agents):
            if j == agent_idx:
                continue
            pB = [self.multi_current_pos[j][0], self.multi_current_pos[j][1]]
            vB = [self.multi_current_vel[j][0], self.multi_current_vel[j][1]]
            combined_r = 2 * ROB_RAD

            # 对友军用 RVO (互避): transl = pA + 0.5*(vA+vB)
            transl_vB_vA = [pA[0] + 0.5 * (vB[0] + vA[0]), pA[1] + 0.5 * (vB[1] + vA[1])]
            dist_BA = distance(pA, pB)
            if combined_r > dist_BA:
                dist_BA_clamped = combined_r
            else:
                dist_BA_clamped = dist_BA
            sin_arg = combined_r / dist_BA_clamped
            sin_arg = min(max(sin_arg, -1.0), 1.0)
            theta_BA = atan2(pB[1] - pA[1], pB[0] - pA[0])
            theta_BAort = asin(sin_arg)
            theta_ort_left = theta_BA + theta_BAort
            theta_ort_right = theta_BA - theta_BAort
            bound_left = [cos(theta_ort_left), sin(theta_ort_left)]
            bound_right = [cos(theta_ort_right), sin(theta_ort_right)]
            RVO_BA = [transl_vB_vA, bound_left, bound_right, dist_BA, combined_r]
            RVO_BA_all.append(RVO_BA)

            cone = self.compute_vo_cone(pA, vA, pB, [0.5 * (vB[0] + vA[0]), 0.5 * (vB[1] + vA[1])], combined_r)
            cones_info.append(cone)

        # ---- 障碍物 ----
        for obs_idx, obstacle in enumerate(ws_model['circular_obstacles']):
            pB = obstacle['position']
            vB = self.obs_vel[obs_idx]
            radius = obstacle['radius']
            rad = radius * OVER_APPROX_C2S
            combined_r = rad + ROB_RAD

            transl_vB_vA = [pA[0] + vB[0], pA[1] + vB[1]]
            dist_BA = distance(pA, pB)
            if combined_r > dist_BA:
                dist_BA_clamped = combined_r
            else:
                dist_BA_clamped = dist_BA
            sin_arg = combined_r / dist_BA_clamped
            sin_arg = min(max(sin_arg, -1.0), 1.0)
            theta_BA = atan2(pB[1] - pA[1], pB[0] - pA[0])
            theta_BAort = asin(sin_arg)
            theta_ort_left = theta_BA + theta_BAort
            theta_ort_right = theta_BA - theta_BAort
            bound_left = [cos(theta_ort_left), sin(theta_ort_left)]
            bound_right = [cos(theta_ort_right), sin(theta_ort_right)]
            RVO_BA = [transl_vB_vA, bound_left, bound_right, dist_BA, combined_r]
            RVO_BA_all.append(RVO_BA)

            cone = self.compute_vo_cone(pA, vA, pB, vB, combined_r)
            cones_info.append(cone)

        return RVO_BA_all, cones_info

    def encode_vo2d_features(self, agent_idx, pA, vA, cones_info):
        """
        方案三（1）：将 VO 锥集合编码为 6 维紧凑特征。
        Returns:
            features (list, length=6):
                0: vo_invasion_score  [0,1] 速度穿透锥的程度 (越深越大)
                1: vo_min_ttc_norm    [0,1] min(TTC) 的归一化指标 1/(1+TTC)，TTC越小值越大
                2: vo_active_count    [0,1] 活跃锥数量 / 总锥数 (归一化)
                3: vo_escape_dir_x    [-1,1] 建议逃离方向的 x 分量 (单位向量)
                4: vo_escape_dir_y    [-1,1] 建议逃离方向的 y 分量
                5: vo_total_occ_angle [0,1] 所有活跃锥在 2π 中占据的总半角和 / π
        """
        if not cones_info:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # 1) vo_invasion_score: 对每个"在锥内"的锥，累积 (half_angle - dist_vo_edge) / half_angle
        invasion = 0.0
        count_in = 0
        for cone in cones_info:
            if cone['in_vo'] and cone['half_angle'] > 1e-4:
                depth = (cone['half_angle'] - cone['dist_vo_edge']) / cone['half_angle']
                invasion = max(invasion, depth)
                count_in += 1
        invasion = float(np.clip(invasion, 0.0, 1.0))

        # 2) vo_min_ttc_norm
        ttcs = [c['ttc'] for c in cones_info if c['distance'] < self.vo_prox_threshold]
        if not ttcs:
            ttcs = [np.inf]
        min_ttc = min(ttcs)
        if min_ttc == np.inf:
            min_ttc_norm = 0.0
        else:
            min_ttc_norm = float(np.clip(1.0 / (1.0 + min_ttc), 0.0, 1.0))
        # 缓存最小 TTC，供奖励 mu9 使用
        self.vo_min_ttc_each[agent_idx] = min_ttc

        # 3) vo_active_count
        n_total = len(cones_info)
        n_active = sum(1 for c in cones_info if c['distance'] < self.vo_prox_threshold)
        active_count = float(n_active / max(n_total, 1))

        # 4,5) vo_escape_dir: 所有活跃锥中心的反方向加权和 (权重 = 1/(1+distance))
        ex, ey = 0.0, 0.0
        any_active = False
        for cone in cones_info:
            if cone['distance'] < self.vo_prox_threshold:
                w = 1.0 / (1.0 + cone['distance'])
                # 方向: 锥中心指向"远离" => 加 π
                theta = cone['center_theta'] + PI
                ex += w * cos(theta)
                ey += w * sin(theta)
                any_active = True
        if any_active and (abs(ex) > 1e-6 or abs(ey) > 1e-6):
            nrm = sqrt(ex * ex + ey * ey)
            escape_x = float(ex / nrm)
            escape_y = float(ey / nrm)
        else:
            # 没有危险方向，默认朝目标方向
            if agent_idx != self.num_agents - 1:
                target = self.multi_current_pos[-1]
                dx = target[0] - pA[0]
                dy = target[1] - pA[1]
                nrm = sqrt(dx * dx + dy * dy) + 1e-6
                escape_x = float(dx / nrm)
                escape_y = float(dy / nrm)
            else:
                escape_x, escape_y = 0.0, 0.0
        escape_x = float(np.clip(escape_x, -1.0, 1.0))
        escape_y = float(np.clip(escape_y, -1.0, 1.0))

        # 6) vo_total_occ_angle
        total_half = sum(c['half_angle'] for c in cones_info if c['distance'] < self.vo_prox_threshold)
        total_occ = float(np.clip(total_half / PI, 0.0, 1.0))

        return [invasion, min_ttc_norm, active_count, escape_x, escape_y, total_occ]

    # =========================================================
    # 方案三（2）：VO 可行域投影 (Post-Action Projection)
    # =========================================================

    def project_velocity_to_vo_feasible(self, pA, v_desired, RVO_BA_all):
        """
        将期望速度 v_desired 投影到 VO 锥可行域外。
        若 v_desired 已可行，原样返回；否则找到距离 v_desired 最近的可行速度。

        复用 RVO.py intersect() 的思想，但做输入输出简化：
        - 搜索角度分辨率更细 (step 0.05 rad ≈ 2.9°)
        - 搜索半径分辨率根据 v_max 自适应

        Returns:
            v_proj (np.ndarray, shape=(2,)): 投影后的速度
            correction_mag (float): ||v_proj - v_desired|| 修正幅度
            was_projected (bool): 是否真的做了投影
        """
        norm_v = np.linalg.norm(v_desired)
        if norm_v < 1e-6:
            # 期望静止：直接判断静止是否安全
            suit = True
            for RVO_BA in RVO_BA_all:
                p_0 = RVO_BA[0]
                left = RVO_BA[1]
                right = RVO_BA[2]
                dif = [0.0 + pA[0] - p_0[0], 0.0 + pA[1] - p_0[1]]
                theta_dif = atan2(dif[1], dif[0])
                theta_right = atan2(right[1], right[0])
                theta_left = atan2(left[1], left[0])
                if in_between(theta_right, theta_dif, theta_left):
                    suit = False
                    break
            if suit:
                return np.array([0.0, 0.0]), 0.0, False
            # 停在原地也不安全，用很小的逃逸速度
            # 搜索一圈方向找最接近0向量的可行解
            suitable_V = []
            for theta in np.arange(0, 2 * PI, 0.1):
                for rad in [0.01, 0.02, 0.05]:
                    new_v = [rad * cos(theta), rad * sin(theta)]
                    ok = True
                    for RVO_BA in RVO_BA_all:
                        p_0 = RVO_BA[0]
                        left = RVO_BA[1]
                        right = RVO_BA[2]
                        dif = [new_v[0] + pA[0] - p_0[0], new_v[1] + pA[1] - p_0[1]]
                        theta_dif = atan2(dif[1], dif[0])
                        theta_right = atan2(right[1], right[0])
                        theta_left = atan2(left[1], left[0])
                        if in_between(theta_right, theta_dif, theta_left):
                            ok = False
                            break
                    if ok:
                        suitable_V.append(new_v)
                        break
            if suitable_V:
                best = min(suitable_V, key=lambda v: np.linalg.norm(np.array(v) - v_desired))
                best = np.array(best)
                return best, float(np.linalg.norm(best - v_desired)), True
            # 万不得已，返回原向量（后续还有碰撞检测兜底）
            return np.array(v_desired), 0.0, False

        # 先判断 v_desired 是否已经可行
        vA = list(v_desired)
        suit = True
        for RVO_BA in RVO_BA_all:
            p_0 = RVO_BA[0]
            left = RVO_BA[1]
            right = RVO_BA[2]
            dif = [vA[0] + pA[0] - p_0[0], vA[1] + pA[1] - p_0[1]]
            theta_dif = atan2(dif[1], dif[0])
            theta_right = atan2(right[1], right[0])
            theta_left = atan2(left[1], left[0])
            if in_between(theta_right, theta_dif, theta_left):
                suit = False
                break
        if suit:
            return np.array(v_desired), 0.0, False

        # 需要投影：网格搜索最近可行速度
        suitable_V = []
        # 半径范围: 0 ~ norm_v + 0.2*v_max (允许稍微加速避碰)
        r_max = norm_v + 0.2 * self.v_max
        r_steps = max(int(norm_v / 0.02) + 2, 5)
        for theta in np.arange(0, 2 * PI, 0.06):  # ≈ 3.4° / step
            for rad in np.linspace(0.0, r_max, r_steps):
                if rad < 1e-5:
                    continue
                new_v = [rad * cos(theta), rad * sin(theta)]
                ok = True
                for RVO_BA in RVO_BA_all:
                    p_0 = RVO_BA[0]
                    left = RVO_BA[1]
                    right = RVO_BA[2]
                    dif = [new_v[0] + pA[0] - p_0[0], new_v[1] + pA[1] - p_0[1]]
                    theta_dif = atan2(dif[1], dif[0])
                    theta_right = atan2(right[1], right[0])
                    theta_left = atan2(left[1], left[0])
                    if in_between(theta_right, theta_dif, theta_left):
                        ok = False
                        break
                if ok:
                    suitable_V.append(new_v)
                    break  # 该方向最近的可行半径已找到，下一个方向

        if suitable_V:
            v_list = [np.array(v) for v in suitable_V]
            # 选 L2 距离 v_desired 最近的
            dists = [np.linalg.norm(v - v_desired) for v in v_list]
            best_idx = int(np.argmin(dists))
            best = v_list[best_idx]
            corr = float(dists[best_idx])
            return best, corr, True

        # 找不到严格可行速度，选择"侵入程度最浅"的
        unsuitable_scores = []
        rad_range = np.linspace(0.0, r_max, 5)
        theta_range = np.arange(0, 2 * PI, 0.1)
        for theta in theta_range:
            for rad in rad_range:
                if rad < 1e-5:
                    continue
                new_v = np.array([rad * cos(theta), rad * sin(theta)])
                # 侵入深度 (最大的锥侵入角度, 越小越好)
                max_depth = 0.0
                for RVO_BA in RVO_BA_all:
                    p_0 = RVO_BA[0]
                    left = RVO_BA[1]
                    right = RVO_BA[2]
                    dif = [new_v[0] + pA[0] - p_0[0], new_v[1] + pA[1] - p_0[1]]
                    theta_dif = atan2(dif[1], dif[0])
                    theta_right = atan2(right[1], right[0])
                    theta_left = atan2(left[1], left[0])
                    if in_between(theta_right, theta_dif, theta_left):
                        # 侵入角度深度 = 半角(取平均) - 到边界距离 (近似)
                        cL = atan2(left[1], left[0])
                        cR = atan2(right[1], right[0])
                        half = abs(cL - cR) / 2
                        if half > PI / 2:
                            half = PI - half
                        max_depth = max(max_depth, half + 0.01)
                WT = 0.2
                score = (max_depth + 1e-3) * WT + np.linalg.norm(new_v - v_desired)
                unsuitable_scores.append((score, new_v))

        best = min(unsuitable_scores, key=lambda x: x[0])[1]
        corr = float(np.linalg.norm(best - v_desired))
        return best, corr, True

    def compute_pred_avoid_flag(self, agent_idx, cones_info, desired_action_dir):
        """
        方案三（3）mu9 判断条件：
          min_TTC < TTC_threshold  (已在 vo_min_ttc_each 缓存)
          AND 动作方向 与 最近VO锥中心 方向差 > 90° (远离)
        """
        min_ttc = self.vo_min_ttc_each[agent_idx]
        if min_ttc >= self.TTC_threshold or min_ttc == np.inf:
            return False

        # 找最近的 (TTC 最小的) 活跃锥
        nearest_cone = None
        nearest_ttc = np.inf
        for c in cones_info:
            if c['ttc'] < nearest_ttc:
                nearest_ttc = c['ttc']
                nearest_cone = c

        if nearest_cone is None:
            return False

        # 锥中心方向 (从 agent 指向 VO 内部的方向 = 危险方向)
        danger_dir = np.array([cos(nearest_cone['center_theta']), sin(nearest_cone['center_theta'])])
        # 动作方向 (若太小，用速度方向)
        if np.linalg.norm(desired_action_dir) < 1e-6:
            vA = self.multi_current_vel[agent_idx]
            if np.linalg.norm(vA) < 1e-6:
                return False
            action_dir = np.array(vA) / (np.linalg.norm(vA) + 1e-6)
        else:
            action_dir = np.array(desired_action_dir) / (np.linalg.norm(desired_action_dir) + 1e-6)

        # 夹角 > 90° 等价于点积 < 0
        dot = float(np.dot(action_dir, danger_dir))
        if dot < 0.0:
            return True
        return False

    # =========================================================
    # 环境核心接口
    # =========================================================

    def get_ws_model(self):
        ws_model = {
            'robot_radius': 0.1,
            'circular_obstacles': [
                {'position': obs.position, 'radius': obs.radius} for obs in self.obstacles
            ]
        }
        return ws_model

    def reset(self):
        SEED = int(time.time() * 1000) % 1000
        random.seed(SEED)
        np.random.seed(SEED)
        self.multi_current_pos = []
        self.multi_current_vel = []
        self.history_positions = [[] for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            if i != self.num_agents - 1:
                self.multi_current_pos.append(np.random.uniform(low=0.1, high=0.4, size=(2,)))
            else:
                self.multi_current_pos.append(np.random.uniform(low=1.3, high=1.8, size=(2,)))
            self.multi_current_vel.append(np.zeros(2))

        # 重置每步缓存
        self.vo_correction_magnitudes = [0.0] * self.num_agents
        self.vo_pred_avoid_flags = [False] * self.num_agents
        self.vo_min_ttc_each = [np.inf] * self.num_agents

        self.update_lasers_isCollied_wrapper()
        multi_obs = self.get_multi_obs()
        return multi_obs

    def step(self, actions):
        """
        方案三（2）：在动作执行前先做 VO 可行域投影。

        流程：
          1) 对每个 hunter，计算 v_desired = v + a * Δt
          2) 做 VO 可行域投影 → v_proj，记录 correction_magnitude
          3) 反解出"实际被执行的加速度" a_proj = (v_proj - v) / Δt
          4) 用 a_proj 更新状态 (位置速度障碍物)
          5) 计算 mu8/mu9 奖励
        """
        last_d2target = []
        self.last_pos = [np.copy(pos) for pos in self.multi_current_pos]

        # 重置每步 VO 相关缓存
        self.vo_correction_magnitudes = [0.0] * self.num_agents
        self.vo_pred_avoid_flags = [False] * self.num_agents
        self.vo_min_ttc_each = [np.inf] * self.num_agents

        projected_actions = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            a_i = np.array(actions[i], dtype=float)

            # 目标 (最后一个 agent) 不做 VO 投影（它是逃避方）
            if i == self.num_agents - 1:
                projected_actions.append(a_i.copy())
                continue

            # 1) 期望速度
            v_desired = vel + a_i * self.time_step
            # 预裁剪速度范围 (先限制 v_max，投影后再最终限制一次)
            vm = np.linalg.norm(v_desired)
            if vm > self.v_max:
                v_desired = v_desired / vm * self.v_max

            # 2) 收集所有 VO 锥并投影
            pA = list(pos)
            RVO_BA_all, cones_info = self.collect_all_vo_cones(i, pA, list(v_desired))
            v_proj, corr_mag, was_proj = self.project_velocity_to_vo_feasible(pA, v_desired, RVO_BA_all)

            self.vo_correction_magnitudes[i] = corr_mag

            # mu9 判断
            # 期望动作的方向 (a_i 方向近似动作方向)
            self.vo_pred_avoid_flags[i] = self.compute_pred_avoid_flag(i, cones_info, np.array(a_i))

            # 3) 反解 a_proj
            a_proj = (v_proj - np.array(vel)) / self.time_step
            # 裁剪加速度范围
            a_norm = np.linalg.norm(a_proj)
            if a_norm > self.a_max:
                a_proj = a_proj / a_norm * self.a_max
            projected_actions.append(a_proj)

        # 4) 用投影后的动作真正更新状态 (原 step 逻辑)
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            if i != self.num_agents - 1:
                pos_taget = self.multi_current_pos[-1]
                last_d2target.append(np.linalg.norm(pos - pos_taget))

            a_i = projected_actions[i]
            self.multi_current_vel[i][0] += a_i[0] * self.time_step
            self.multi_current_vel[i][1] += a_i[1] * self.time_step
            vel_magnitude = np.linalg.norm(self.multi_current_vel[i])
            if i != self.num_agents - 1:
                if vel_magnitude >= self.v_max:
                    self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max
            else:
                if vel_magnitude >= self.v_max_e:
                    self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max_e

            self.multi_current_pos[i][0] += self.multi_current_vel[i][0] * self.time_step
            self.multi_current_pos[i][1] += self.multi_current_vel[i][1] * self.time_step

        # Update obstacle positions
        for obs in self.obstacles:
            obs.position += obs.velocity * self.time_step
            for dim in [0, 1]:
                if obs.position[dim] - obs.radius < 0:
                    obs.position[dim] = obs.radius
                    obs.velocity[dim] *= -1
                elif obs.position[dim] + obs.radius > self.length:
                    obs.position[dim] = self.length - obs.radius
                    obs.velocity[dim] *= -1

        Collided = self.update_lasers_isCollied_wrapper()
        rewards, dones = self.cal_rewards_dones(Collided, last_d2target, self.last_pos)
        multi_next_obs = self.get_multi_obs()
        multi_pos = self.multi_current_pos

        return multi_next_obs, rewards, dones, Collided, multi_pos

    def get_multi_obs(self):
        total_obs = []
        single_obs = []
        S_evade_d = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            S_uavi = [
                pos[0] / self.length,
                pos[1] / self.length,
                vel[0] / self.v_max,
                vel[1] / self.v_max
            ]  # dim 4
            S_team = []
            S_target = []
            for j in range(self.num_agents):
                if j != i and j != self.num_agents - 1:
                    pos_other = self.multi_current_pos[j]
                    S_team.extend([pos_other[0] / self.length, pos_other[1] / self.length])
                elif j == self.num_agents - 1:
                    pos_target = self.multi_current_pos[j]
                    d = np.linalg.norm(pos - pos_target)
                    theta = np.arctan2(pos_target[1] - pos[1], pos_target[0] - pos[0])
                    S_target.extend([d / np.linalg.norm(2 * self.length), theta])
                    if i != self.num_agents - 1:
                        S_evade_d.append(d / np.linalg.norm(2 * self.length))

            S_obser = self.multi_current_lasers[i]  # dim 16

            # ===== 方案三（1）：追加 VO 2D 立体锥编码 =====
            pA = list(pos)
            vA = list(vel)
            _RVO_all, cones_info = self.collect_all_vo_cones(i, pA, vA)
            vo_features = self.encode_vo2d_features(i, pA, vA, cones_info)  # dim 6

            if i != self.num_agents - 1:
                # hunter: 4 + 4 + 16 + 2 + 6 = 32
                single_obs = [S_uavi, S_team, S_obser, S_target, vo_features]
            else:
                # target: 4 + 16 + 3 + 6 = 29
                single_obs = [S_uavi, S_obser, S_evade_d, vo_features]
            _single_obs = list(itertools.chain(*single_obs))
            total_obs.append(_single_obs)

        return total_obs

    def VO_Plot(self, X, V_current, ws_model, FLAG=True, ax=None):
        if not FLAG or ax is None:
            return
        ROB_RAD = ws_model['robot_radius'] - 0.08
        for i in range(len(X)):
            pA = [X[i][0], X[i][1]]
            vA = [V_current[i][0], V_current[i][1]]
            for j in range(len(X)):
                if i != j:
                    pB = [X[j][0], X[j][1]]
                    vB = [V_current[j][0], V_current[j][1]]
                    dist_BA = np.linalg.norm(np.array(pB) - np.array(pA))
                    theta_BA = np.arctan2(pB[1] - pA[1], pB[0] - pA[0])
                    if 2 * ROB_RAD > dist_BA:
                        dist_BA = 2 * ROB_RAD
                    theta_BAort = np.arcsin(2 * ROB_RAD / dist_BA)
                    theta_ort_left = theta_BA + theta_BAort
                    theta_ort_right = theta_BA - theta_BAort
            for obs_idx, obstacle in enumerate(ws_model['circular_obstacles']):
                pB = obstacle['position']
                vB = self.obs_vel[obs_idx]
                radius = obstacle['radius']
                transl_vB_vA = [pA[0] + vB[0], pA[1] + vB[1]]
                dist_BA = np.linalg.norm(np.array(pB) - np.array(pA))
                theta_BA = np.arctan2(pB[1] - pA[1], pB[0] - pA[0])
                OVER_APPROX_C2S = 1.3
                rad = radius * OVER_APPROX_C2S
                if (rad + ROB_RAD) > dist_BA:
                    dist_BA = rad + ROB_RAD
                theta_BAort = np.arcsin((rad + ROB_RAD) / dist_BA)
                theta_ort_left = theta_BA + theta_BAort
                theta_ort_right = theta_BA - theta_BAort
                ax.plot(
                    [pA[0], transl_vB_vA[0] + np.cos(theta_ort_left)],
                    [pA[1], transl_vB_vA[1] + np.sin(theta_ort_left)],
                    color='red', linestyle='--', alpha=0.7
                )
                ax.plot(
                    [pA[0], transl_vB_vA[0] + np.cos(theta_ort_right)],
                    [pA[1], transl_vB_vA[1] + np.sin(theta_ort_right)],
                    color='red', linestyle='--', alpha=0.7
                )
                ax.scatter(pB[0], pB[1], color='red', s=50, label='Obstacle')
                ax.arrow(
                    pB[0], pB[1], vB[0], vB[1],
                    head_width=0.05, head_length=0.1, fc='red', ec='red', alpha=0.8
                )
            ax.arrow(
                pA[0], pA[1], vA[0], vA[1],
                head_width=0.05, head_length=0.1, fc='blue', ec='blue', alpha=0.8
            )
        ax.legend()

    def VO_reward(self, X, V_current, ws_model, dist_threshold=0.4):
        ROB_RAD = ws_model['robot_radius'] - 0.05
        rewards = [0] * len(X)
        for i in range(len(X)):
            pA = [X[i][0], X[i][1]]
            vA = [V_current[i][0], V_current[i][1]]
            RVO_BA_all = []
            for j in range(len(X)):
                if i != j:
                    pB = [X[j][0], X[j][1]]
                    vB = [V_current[j][0], V_current[j][1]]
                    transl_vB_vA = [pA[0] + 0.5 * (vB[0] + vA[0]), pA[1] + 0.5 * (vB[1] + vA[1])]
                    dist_BA = distance(pA, pB)
                    theta_BA = atan2(pB[1] - pA[1], pB[0] - pA[0])
                    if 2 * ROB_RAD > dist_BA:
                        dist_BA = 2 * ROB_RAD
                    theta_BAort = asin(2 * ROB_RAD / dist_BA)
                    theta_ort_left = theta_BA + theta_BAort
                    theta_ort_right = theta_BA - theta_BAort
                    bound_left = [cos(theta_ort_left), sin(theta_ort_left)]
                    bound_right = [cos(theta_ort_right), sin(theta_ort_right)]
                    RVO_BA = [transl_vB_vA, bound_left, bound_right, dist_BA, 2 * ROB_RAD]
                    RVO_BA_all.append(RVO_BA)
            for obs_idx, obstacle in enumerate(ws_model['circular_obstacles']):
                pB = obstacle['position']
                vB = self.obs_vel[obs_idx]
                radius = obstacle['radius']
                transl_vB_vA = [pA[0] + vB[0], pA[1] + vB[1]]
                dist_BA = distance(pA, pB)
                theta_BA = atan2(pB[1] - pA[1], pB[0] - pA[0])
                OVER_APPROX_C2S = 1.3
                rad = radius * OVER_APPROX_C2S
                if (rad + ROB_RAD) > dist_BA:
                    dist_BA = rad + ROB_RAD
                theta_BAort = asin((rad + ROB_RAD) / dist_BA)
                theta_ort_left = theta_BA + theta_BAort
                theta_ort_right = theta_BA - theta_BAort
                bound_left = [cos(theta_ort_left), sin(theta_ort_left)]
                bound_right = [cos(theta_ort_right), sin(theta_ort_right)]
                RVO_BA = [transl_vB_vA, bound_left, bound_right, dist_BA, rad + ROB_RAD]
                RVO_BA_all.append(RVO_BA)
            in_vo = False
            for RVO_BA in RVO_BA_all:
                p_0 = RVO_BA[0]
                left = RVO_BA[1]
                right = RVO_BA[2]
                dif = [vA[0] + pA[0] - p_0[0], vA[1] + pA[1] - p_0[1]]
                theta_dif = atan2(dif[1], dif[0])
                theta_right = atan2(right[1], right[0])
                theta_left = atan2(left[1], left[0])
                if in_between(theta_right, theta_dif, theta_left):
                    in_vo = True
                    if RVO_BA[3] < dist_threshold:
                        rewards[i] = -10
            if not in_vo:
                rewards[i] = 0
        return rewards

    def VO_reward_obs(self, X, V_current, ws_model, dist_threshold=0.4):
        ROB_RAD = ws_model['robot_radius'] + 0.02
        rewards = [0] * len(X)
        for i in range(len(X)):
            pA = [X[i][0], X[i][1]]
            vA = [V_current[i][0], V_current[i][1]]
            RVO_BA_all = []
            for obs_idx, obstacle in enumerate(ws_model['circular_obstacles']):
                pB = obstacle['position']
                vB = self.obs_vel[obs_idx]
                radius = obstacle['radius']
                transl_vB_vA = [pA[0] + vB[0], pA[1] + vB[1]]
                dist_BA = distance(pA, pB)
                theta_BA = atan2(pB[1] - pA[1], pB[0] - pA[0])
                OVER_APPROX_C2S = 1.3
                rad = radius * OVER_APPROX_C2S
                if (rad + ROB_RAD) > dist_BA:
                    dist_BA = rad + ROB_RAD
                theta_BAort = asin((rad + ROB_RAD) / dist_BA)
                theta_ort_left = theta_BA + theta_BAort
                theta_ort_right = theta_BA - theta_BAort
                bound_left = [cos(theta_ort_left), sin(theta_ort_left)]
                bound_right = [cos(theta_ort_right), sin(theta_ort_right)]
                RVO_BA = [transl_vB_vA, bound_left, bound_right, dist_BA, rad + ROB_RAD]
                RVO_BA_all.append(RVO_BA)
            for RVO_BA in RVO_BA_all:
                p_0 = RVO_BA[0]
                left = RVO_BA[1]
                right = RVO_BA[2]
                dif = [vA[0] + pA[0] - p_0[0], vA[1] + pA[1] - p_0[1]]
                theta_dif = atan2(dif[1], dif[0])
                theta_right = atan2(right[1], right[0])
                theta_left = atan2(left[1], left[0])
                theta_center = (theta_left + theta_right) / 2
                if theta_left > theta_right:
                    theta_center = (theta_left + theta_right + 2 * np.pi) / 2
                    if theta_center > np.pi:
                        theta_center -= 2 * np.pi
                angle_diff = abs(theta_dif - theta_center)
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff
                if in_between(theta_right, theta_dif, theta_left):
                    angle_penalty = angle_diff / (np.pi)
                    rewards[i] -= 10 * (angle_penalty)
                else:
                    rewards[i] += 0
        return rewards

    def calculate_expected_collision_time(self, pA, vA, pB, vB, combined_radius):
        delta_p = np.array(pB) - np.array(pA)
        delta_v = np.array(vB) - np.array(vA)
        a = np.dot(delta_v, delta_v)
        b = 2 * np.dot(delta_p, delta_v)
        c = np.dot(delta_p, delta_p) - combined_radius ** 2
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return np.inf
        t1 = (-b + np.sqrt(discriminant)) / (2 * max(a, 1e-8))
        t2 = (-b - np.sqrt(discriminant)) / (2 * max(a, 1e-8))
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
        elif t1 > 0:
            return t1
        elif t2 > 0:
            return t2
        else:
            return np.inf

    def VO_reward_obs_rvo(self, X, V_current, ws_model, dist_threshold=0.4):
        ROB_RAD = ws_model['robot_radius'] + 0.02
        rewards = [0] * len(X)
        a, b, c, d, e, f = 0.2, 0.5, 0.3, 2.4, 5.8, 0.2
        for i in range(len(X)):
            pA = [X[i][0], X[i][1]]
            vA = [V_current[i][0], V_current[i][1]]
            RVO_BA_all = []
            for obs_idx, obstacle in enumerate(ws_model['circular_obstacles']):
                pB = obstacle['position']
                vB = self.obs_vel[obs_idx]
                radius = obstacle['radius']
                transl_vB_vA = [pA[0] + vB[0], pA[1] + vB[1]]
                dist_BA = distance(pA, pB)
                theta_BA = atan2(pB[1] - pA[1], pB[0] - pA[0])
                OVER_APPROX_C2S = 1.3
                rad = radius * OVER_APPROX_C2S
                if (rad + ROB_RAD) > dist_BA:
                    dist_BA = rad + ROB_RAD
                theta_BAort = asin((rad + ROB_RAD) / dist_BA)
                theta_ort_left = theta_BA + theta_BAort
                theta_ort_right = theta_BA - theta_BAort
                bound_left = [cos(theta_ort_left), sin(theta_ort_left)]
                bound_right = [cos(theta_ort_right), sin(theta_ort_right)]
                RVO_BA = [transl_vB_vA, bound_left, bound_right, dist_BA, rad + ROB_RAD]
                RVO_BA_all.append(RVO_BA)
            for RVO_BA in RVO_BA_all:
                p_0 = RVO_BA[0]
                left = RVO_BA[1]
                right = RVO_BA[2]
                dif = [vA[0] + pA[0] - p_0[0], vA[1] + pA[1] - p_0[1]]
                theta_dif = atan2(dif[1], dif[0])
                theta_right = atan2(right[1], right[0])
                theta_left = atan2(left[1], left[0])
                theta_center = (theta_left + theta_right) / 2
                if theta_left > theta_right:
                    theta_center = (theta_left + theta_right + 2 * np.pi) / 2
                    if theta_center > np.pi:
                        theta_center -= 2 * np.pi
                angle_diff = abs(theta_dif - theta_center)
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff
                expected_collision_time = self.calculate_expected_collision_time(pA, vA, pB, vB, ROB_RAD + rad)
                if in_between(theta_right, theta_dif, theta_left):
                    if expected_collision_time > 0.1:
                        rewards[i] += c - d * (expected_collision_time + f) ** -1
                    else:
                        rewards[i] -= e * (expected_collision_time + f) ** -1
                else:
                    if expected_collision_time > 4:
                        rewards[i] = a - b * np.linalg.norm(V_current[i] - [0.1, 0.1])
        return rewards

    def cal_rewards_dones(self, IsCollied, last_d, last_pos):
        dones = [False] * self.num_agents
        rewards = np.zeros(self.num_agents)
        mu1 = 0.9
        mu2 = 0.2
        mu3 = 0.0
        mu4 = 10
        mu5 = 0.1
        d_capture = 0.3
        d_limit = 0.55
        current_positions = np.array(self.multi_current_pos[:self.num_agents - 1])
        target_position = self.multi_current_pos[-1]

        # 1) r_near
        for i in range(3):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            pos_target = self.multi_current_pos[-1]
            v_i = np.linalg.norm(vel)
            dire_vec = pos_target - pos
            d = np.linalg.norm(dire_vec)
            cos_v_d = np.dot(vel, dire_vec) / (v_i * d + 1e-3)
            r_near = abs(2 * v_i / self.v_max) * cos_v_d
            rewards[i] += mu1 * r_near

        # 2) r_safe (collision/laser based)
        for i in range(self.num_agents):
            if IsCollied[i]:
                r_safe = -10
            else:
                lasers = self.multi_current_lasers[i]
                r_safe = (min(lasers) - self.L_sensor - 0.1) / self.L_sensor
            rewards[i] += mu2 * r_safe

        # VO legacy reward (mu5)
        ws_model = self.get_ws_model()
        vo_rewards = self.VO_reward(self.multi_current_pos, self.multi_current_vel, ws_model,
                                    dist_threshold=0.7)
        vo_rewards_float = [float(x) for x in vo_rewards[0:3]]
        rewards[0:3] += mu5 * np.array(vo_rewards_float)

        # ===== 方案三（3）：新增 mu8 / mu9 =====
        # mu8: VO 修正惩罚 (只对 3 个 hunters)
        for i in range(3):
            corr = self.vo_correction_magnitudes[i]
            if corr > 0:
                # 归一化修正量 / v_max
                r_mu8 = self.mu8 * (corr / max(self.v_max, 1e-6))
                rewards[i] += r_mu8

        # mu9: 预测性避碰加分 (只对 3 个 hunters)
        for i in range(3):
            if self.vo_pred_avoid_flags[i]:
                rewards[i] += self.mu9

        # 3) multi-stage's reward for rounding-up-UAVs
        p0 = self.multi_current_pos[0]
        p1 = self.multi_current_pos[1]
        p2 = self.multi_current_pos[2]
        pe = self.multi_current_pos[-1]
        S1 = cal_triangle_S(p0, p1, pe)
        S2 = cal_triangle_S(p1, p2, pe)
        S3 = cal_triangle_S(p2, p0, pe)
        S4 = cal_triangle_S(p0, p1, p2)
        d1 = np.linalg.norm(p0 - pe)
        d2 = np.linalg.norm(p1 - pe)
        d3 = np.linalg.norm(p2 - pe)
        Sum_S = S1 + S2 + S3
        Sum_d = d1 + d2 + d3
        Sum_last_d = sum(last_d)
        rewards[-1] += np.clip(2 * (Sum_d - Sum_last_d), -2, 2)

        if Sum_S > S4 and Sum_d >= d_limit and all(d >= d_capture for d in [d1, d2, d3]):
            r_track = - Sum_d / max([d1, d2, d3])
            rewards[0:3] += mu3 * r_track
        elif Sum_S > S4 and (Sum_d < d_limit or any(d >= d_capture for d in [d1, d2, d3])):
            r_encircle = -1 / 3 * np.log(Sum_S - S4 + 1)
            rewards[0:3] += mu3 * r_encircle
        elif Sum_S == S4 and any(d > d_capture for d in [d1, d2, d3]):
            r_capture = np.exp((Sum_last_d - Sum_d) / (3 * self.v_max))
            rewards[0:3] += mu3 * r_capture

        if Sum_S == S4 and all(d <= d_capture for d in [d1, d2, d3]):
            rewards[0:3] += mu4 * 10
            dones = [True] * self.num_agents

        return rewards, dones

    def update_lasers_isCollied_wrapper(self):
        self.multi_current_lasers = []
        dones = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            current_lasers = [self.L_sensor] * self.num_lasers
            done_obs = []
            for obs in self.obstacles:
                obs_pos = obs.position
                r = obs.radius
                _current_lasers, done = update_lasers(pos, obs_pos, r, self.L_sensor, self.num_lasers, self.length)
                current_lasers = [min(l, cl) for l, cl in zip(_current_lasers, current_lasers)]
                done_obs.append(done)
            done = any(done_obs)
            if done:
                self.multi_current_vel[i] = np.zeros(2)
            self.multi_current_lasers.append(current_lasers)
            dones.append(done)
        return dones

    def render(self):
        plt.clf()
        uav_icon = mpimg.imread('UAV.png')
        for i in range(self.num_agents - 1):
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            self.history_positions[i].append(pos)
            trajectory = np.array(self.history_positions[i])
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.3)
            angle = np.arctan2(vel[1], vel[0])
            t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            icon_size = 0.1
            plt.imshow(uav_icon, transform=t + plt.gca().transData,
                       extent=(-icon_size / 2, icon_size / 2, -icon_size / 2, icon_size / 2))
        plt.scatter(self.multi_current_pos[-1][0], self.multi_current_pos[-1][1], c='r', label='Target')
        self.history_positions[-1].append(copy.deepcopy(self.multi_current_pos[-1]))
        trajectory = np.array(self.history_positions[-1])
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.3)
        for obstacle in self.obstacles:
            circle = plt.Circle(obstacle.position, obstacle.radius, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)
        plt.xlim(-0.1, self.length + 0.1)
        plt.ylim(-0.1, self.length + 0.1)
        plt.draw()
        plt.legend()
        canvas = agg.FigureCanvasAgg(plt.gcf())
        canvas.draw()
        buf = canvas.buffer_rgba()
        image = np.asarray(buf)
        return image

    def render_anime(self, frame_num):
        plt.clf()
        uav_icon = mpimg.imread('UAV.png')
        for i in range(self.num_agents - 1):
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            angle = np.arctan2(vel[1], vel[0])
            self.history_positions[i].append(pos)
            trajectory = np.array(self.history_positions[i])
            for j in range(len(trajectory) - 1):
                color = cm.viridis(j / len(trajectory))
                plt.plot(trajectory[j:j + 2, 0], trajectory[j:j + 2, 1], color=color, alpha=0.7)
            t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            icon_size = 0.1
            plt.imshow(uav_icon, transform=t + plt.gca().transData,
                       extent=(-icon_size / 2, icon_size / 2, -icon_size / 2, icon_size / 2))
        plt.scatter(self.multi_current_pos[-1][0], self.multi_current_pos[-1][1], c='r', label='Target')
        pos_e = copy.deepcopy(self.multi_current_pos[-1])
        self.history_positions[-1].append(pos_e)
        trajectory = np.array(self.history_positions[-1])
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.3)
        for obstacle in self.obstacles:
            circle = plt.Circle(obstacle.position, obstacle.radius, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)
        plt.xlim(-0.1, self.length + 0.1)
        plt.ylim(-0.1, self.length + 0.1)
        plt.draw()

    def close(self):
        plt.close()


class obstacle():
    def __init__(self, length=2, mode='random', index=0, total_obstacles=4):
        if mode == 'random':
            self.position = np.random.uniform(low=0.45, high=length - 0.55, size=(2,))
            angle = np.random.uniform(0, 2 * np.pi)
            speed = 0.025
            self.velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])
            self.radius = np.random.uniform(0.14, 0.18)
        elif mode == "fixed":
            position_offset = np.random.uniform(-0.5, 0.5, size=2)
            velocity_offset = np.random.uniform(-0.04, 0.04, size=2)
            if index % 2 == 0:
                start_x = 0
                start_y = 2 - (index // 2) * (2 / (total_obstacles // 2))
                self.position = np.array([start_x, start_y]) + position_offset
                if index == 1:
                    self.velocity = np.array([0.03, -0.02]) + velocity_offset
                else:
                    self.velocity = np.array([0.02, -0.03]) + velocity_offset
            else:
                start_x = 2
                start_y = (index // 2) * (2 / (total_obstacles // 2))
                self.position = np.array([start_x, start_y]) + position_offset
                self.velocity = np.array([-0.02, 0.03]) + velocity_offset
            self.radius = np.random.uniform(0.15, 0.20)
