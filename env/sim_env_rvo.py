import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.cm as cm
import matplotlib.image as mpimg
from gymnasium import spaces
from torchaudio.functional import speed

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
    def __init__(self,length=2,num_obstacle=4,num_agents=4):
        self.length = length # length of boundary
        self.num_obstacle = num_obstacle # number of obstacles
        self.num_agents = num_agents
        self.time_step = 0.5 # update time step
        self.v_max = 0.12
        self.v_max_e = 0.03
        self.a_max = 0.06
        self.a_max_e = 0.04
        self.L_sensor = 0.2
        self.num_lasers = 16 # num of laserbeams
        self.multi_current_lasers = [[self.L_sensor for _ in range(self.num_lasers)] for _ in range(self.num_agents)]
        self.agents = ['agent_0','agent_1','agent_2','target']
        self.info = np.random.get_state() # get seed
        # self.obstacles = [obstacle() for _ in range(self.num_obstacle)]
        self.obstacles = [
            obstacle(mode='random', index=i, total_obstacles=self.num_obstacle)
            for i in range(self.num_obstacle)
        ]
        self.history_positions = [[] for _ in range(num_agents)]
        self.obs_vel = [obs.velocity for obs in self.obstacles]  # 初始化障碍物速度
        self.last_pos = [np.zeros(2) for _ in range(num_agents)]  # 初始化为零向量


        self.action_space = {
            'agent_0': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'agent_1': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'agent_2': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'target': spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
            } # action represents [a_x,a_y]
        self.observation_space = {
            'agent_0': spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            'agent_1': spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            'agent_2': spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            'target': spaces.Box(low=-np.inf, high=np.inf, shape=(23,))
        }

    def get_ws_model(self):
        """
        构造 ws_model，包含机器人半径和所有障碍物的信息。
        """
        ws_model = {
            'robot_radius': 0.1,  # 假设机器人半径固定为 0.1
            'circular_obstacles': [
                {'position': obs.position, 'radius': obs.radius} for obs in self.obstacles
            ]
        }
        return ws_model
    def reset(self):
        # SEED = random.randint(1,1000)
        SEED = int(time.time() * 1000) % 1000
        random.seed(SEED)
        np.random.seed(SEED)
        self.multi_current_pos = []
        self.multi_current_vel = []
        self.history_positions = [[] for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            if i != self.num_agents - 1: # if not target
                self.multi_current_pos.append(np.random.uniform(low=0.1,high=0.4,size=(2,)))
            else: # for target
                # self.multi_current_pos.append(np.array([1.5,1.5]))
                # self.multi_current_pos.append(np.array([0.5,1.75]))
                self.multi_current_pos.append(np.random.uniform(low=1.3, high=1.8, size=(2,)))
            self.multi_current_vel.append(np.zeros(2)) # initial velocity = [0,0]
            # print(self.multi_current_pos[i])

        # update lasers
        self.update_lasers_isCollied_wrapper()
        ## multi_obs is list of agent_obs, state is multi_obs after flattenned
        multi_obs = self.get_multi_obs()

        return multi_obs

    def compute_acceleration(self, actions):
        """
        根据动力学模型计算无人机的加速度。
        Args:
            actions: 控制输入 [φ, u_th]，即滚转角和油门。
        Returns:
            accelerations: 计算出的加速度列表 [[a_x, a_y], ...]
        """
        delta_t = 0.01  # 时间步长
        Td = 0.04  # 传感器延迟时间
        a1, a2, b1 = -0.1, -0.2, 0.5  # 动力学模型参数

        accelerations = []
        for action in actions:
            phi, u_th = action  # 提取控制输入

            # 动力学模型矩阵
            A = np.array([
                [1, delta_t, 0, 0],
                [0, 1 - delta_t / Td, delta_t / Td, 0],
                [0, 0, 1, delta_t],
                [0, 0, a1 * delta_t, 1 + a2 * delta_t]
            ])
            B = np.array([0, 0, 0, b1 * delta_t]).reshape(-1, 1)

            # 初始状态（仅使用速度和加速度部分）
            state_x = np.array([0, 0, 0, 0])  # 假设初始速度和加速度为零
            input_x = np.array([phi])  # 控制输入

            # 解算加速度
            state_next = A @ state_x + B @ input_x
            a_x = state_next[2]  # 提取加速度
            a_y = u_th  # 将油门直接映射为 y 轴加速度（假设简单映射）

            accelerations.append([a_x, a_y])

        return accelerations

    def step(self, actions):

        last_d2target = []
        self.last_pos = [np.copy(pos) for pos in self.multi_current_pos]
        # print(actions)
        # time.sleep(0.1)
        for i in range(self.num_agents):

            pos = self.multi_current_pos[i]
            if i != self.num_agents - 1:
                pos_taget = self.multi_current_pos[-1]
                last_d2target.append(np.linalg.norm(pos - pos_taget))

            self.multi_current_vel[i][0] += actions[i][0] * self.time_step
            self.multi_current_vel[i][1] += actions[i][1] * self.time_step
            vel_magnitude = np.linalg.norm(self.multi_current_vel)
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
            # Check for boundary collisions and adjust velocities
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
        # sequence above can't be disrupted
        multi_pos = self.multi_current_pos

        return multi_next_obs, rewards, dones, Collided, multi_pos

    def step_2(self, actions):
        """
        根据动力学模型计算加速度，并更新无人机的状态。
        Args:
            actions: 控制输入 [φ, u_th]。
        Returns:
            multi_next_obs: 下一步的观测空间。
            rewards: 奖励值。
            dones: 是否结束。
            Collided: 碰撞信息。
            multi_pos: 当前无人机位置。
        """
        # 根据动力学模型计算加速度
        accelerations = self.compute_acceleration(actions)

        last_d2target = []
        self.last_pos = [np.copy(pos) for pos in self.multi_current_pos]

        for i in range(self.num_agents - 1):
            pos = self.multi_current_pos[i]
            if i != self.num_agents - 1:
                pos_taget = self.multi_current_pos[-1]
                last_d2target.append(np.linalg.norm(pos - pos_taget))

            # 使用计算出的加速度更新速度
            self.multi_current_vel[i][0] += accelerations[i][0] * self.time_step
            self.multi_current_vel[i][1] += accelerations[i][1] * self.time_step

            # 限制速度范围
            vel_magnitude = np.linalg.norm(self.multi_current_vel[i])
            if i != self.num_agents - 1:
                if vel_magnitude >= self.v_max:
                    self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max
            else:
                if vel_magnitude >= self.v_max_e:
                    self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max_e

            # 更新位置
            self.multi_current_pos[i][0] += self.multi_current_vel[i][0] * self.time_step
            self.multi_current_pos[i][1] += self.multi_current_vel[i][1] * self.time_step

        # 更新障碍物位置
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

    def test_multi_obs(self):
        total_obs = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            S_uavi = [
                pos[0]/self.length,
                pos[1]/self.length,
                vel[0]/self.v_max,
                vel[1]/self.v_max
            ]
            total_obs.append(S_uavi)
        return total_obs

    def get_multi_obs(self):
        total_obs = []
        single_obs = []
        S_evade_d = [] # dim 3 only for target
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            S_uavi = [
                pos[0]/self.length,
                pos[1]/self.length,
                vel[0]/self.v_max,
                vel[1]/self.v_max
            ] # dim 4
            S_team = [] # dim 4 for 3 agents 1 target
            S_target = [] # dim 2
            for j in range(self.num_agents):
                if j != i and j != self.num_agents - 1:
                    pos_other = self.multi_current_pos[j]
                    S_team.extend([pos_other[0]/self.length,pos_other[1]/self.length])
                elif j == self.num_agents - 1:
                    pos_target = self.multi_current_pos[j]
                    d = np.linalg.norm(pos - pos_target)
                    theta = np.arctan2(pos_target[1]-pos[1], pos_target[0]-pos[0])
                    S_target.extend([d/np.linalg.norm(2*self.length), theta])
                    if i != self.num_agents - 1:
                        S_evade_d.append(d/np.linalg.norm(2*self.length))

            S_obser = self.multi_current_lasers[i] # dim 16

            if i != self.num_agents - 1:
                single_obs = [S_uavi,S_team,S_obser,S_target]
            else:
                single_obs = [S_uavi,S_obser,S_evade_d]
            _single_obs = list(itertools.chain(*single_obs))
            total_obs.append(_single_obs)

        return total_obs

    def VO_Plot(self, X, V_current, ws_model, FLAG=True, ax=None):
        """
        在已有的图像上绘制所有 UAV 和障碍物的 VO 区域。

        Args:
            X: List of UAV positions, e.g., [[x1, y1], [x2, y2], ...]
            V_current: List of current velocities, e.g., [[vx1, vy1], [vx2, vy2], ...]
            ws_model: Workspace model containing robot radius and obstacle information.
            FLAG: Boolean flag to control whether to draw VO regions.
            ax: Matplotlib Axes 对象，用于在已有的图上绘制 VO 区域。
        """
        if not FLAG or ax is None:
            return

        ROB_RAD = ws_model['robot_radius'] - 0.08

        for i in range(len(X)):
            pA = [X[i][0], X[i][1]]  # 当前 UAV 位置
            vA = [V_current[i][0], V_current[i][1]]  # 当前 UAV 速度

            # 绘制 UAV 的 VO 区域
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

                    # # 绘制 VO 边界
                    # ax.plot(
                    #     [pA[0], pA[0] + np.cos(theta_ort_left)],
                    #     [pA[1], pA[1] + np.sin(theta_ort_left)],
                    #     color='orange', linestyle='--', alpha=0.7
                    # )
                    # ax.plot(
                    #     [pA[0], pA[0] + np.cos(theta_ort_right)],
                    #     [pA[1], pA[1] + np.sin(theta_ort_right)],
                    #     color='orange', linestyle='--', alpha=0.7
                    # )

            # 绘制障碍物的 VO 区域
            for obs_idx, obstacle in enumerate(ws_model['circular_obstacles']):
                pB = obstacle['position']  # 障碍物位置
                vB = self.obs_vel[obs_idx]  # 障碍物速度
                radius = obstacle['radius']  # 障碍物半径

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

                # 绘制障碍物的 VO 边界
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

                # 绘制障碍物位置和速度向量
                ax.scatter(pB[0], pB[1], color='red', s=50, label='Obstacle')
                ax.arrow(
                    pB[0], pB[1], vB[0], vB[1],
                    head_width=0.05, head_length=0.1, fc='red', ec='red', alpha=0.8
                )

            # 绘制 UAV 的速度向量
            ax.arrow(
                pA[0], pA[1], vA[0], vA[1],
                head_width=0.05, head_length=0.1, fc='blue', ec='blue', alpha=0.8
            )

        ax.legend()

    # def VO_reward(self, X, V_current, ws_model, dist_threshold=0.4):
    #     """
    #     Compute reward for each robot based on VO region and distance to obstacles.
    #
    #     Args:
    #         X: List of robot positions, e.g., [[x1, y1], [x2, y2], ...]
    #         V_current: List of current velocities, e.g., [[vx1, vy1], [vx2, vy2], ...]
    #         ws_model: Workspace model containing robot radius and obstacle information.
    #         dist_threshold: Distance threshold for applying negative reward.
    #
    #     Returns:
    #         List of rewards for each robot, e.g., [-1, 0, 0, -1, ...]
    #     """
    #     ROB_RAD = ws_model['robot_radius']-0.02
    #     rewards = [0] * len(X)  # Initialize rewards for all robots
    #
    #     for i in range(len(X)):
    #         pA = [X[i][0], X[i][1]]  # Current robot position
    #         vA = [V_current[i][0], V_current[i][1]]  # Current robot velocity
    #         RVO_BA_all = []
    #
    #         # Compute RVO regions for other robots
    #         for j in range(len(X)):
    #             if i != j:
    #                 pB = [X[j][0], X[j][1]]
    #                 vB = [V_current[j][0], V_current[j][1]]
    #
    #                 # Compute translational velocity and VO bounds
    #                 transl_vB_vA = [pA[0] + 0.5 * (vB[0] + vA[0]), pA[1] + 0.5 * (vB[1] + vA[1])]
    #                 dist_BA = distance(pA, pB)
    #                 theta_BA = atan2(pB[1] - pA[1], pB[0] - pA[0])
    #
    #                 if 2 * ROB_RAD > dist_BA:
    #                     dist_BA = 2 * ROB_RAD
    #
    #                 theta_BAort = asin(2 * ROB_RAD / dist_BA)
    #                 theta_ort_left = theta_BA + theta_BAort
    #                 theta_ort_right = theta_BA - theta_BAort
    #                 bound_left = [cos(theta_ort_left), sin(theta_ort_left)]
    #                 bound_right = [cos(theta_ort_right), sin(theta_ort_right)]
    #
    #                 RVO_BA = [transl_vB_vA, bound_left, bound_right, dist_BA, 2 * ROB_RAD]
    #                 RVO_BA_all.append(RVO_BA)
    #
    #         # Compute RVO regions for circular obstacles
    #         # for obstacle in ws_model['circular_obstacles']:
    #         for obs_idx, obstacle in enumerate(ws_model['circular_obstacles']):
    #             pB = obstacle['position']  # 障碍物位置
    #             # vB = [0, 0]  # 障碍物静止
    #             vB = self.obs_vel[obs_idx]
    #             radius = obstacle['radius']  # 障碍物半径
    #             # print(f"Obstacle position: {pB}, radius: {radius}")
    #
    #
    #             transl_vB_vA = [pA[0] + vB[0], pA[1] + vB[1]]
    #             dist_BA = distance(pA, pB)
    #             theta_BA = atan2(pB[1] - pA[1], pB[0] - pA[0])
    #
    #             OVER_APPROX_C2S = 1.3
    #             rad = radius * OVER_APPROX_C2S  # 使用障碍物的随机半径
    #
    #             if (rad + ROB_RAD) > dist_BA:
    #                 dist_BA = rad + ROB_RAD
    #
    #             theta_BAort = asin((rad + ROB_RAD) / dist_BA)
    #             theta_ort_left = theta_BA + theta_BAort
    #             theta_ort_right = theta_BA - theta_BAort
    #             bound_left = [cos(theta_ort_left), sin(theta_ort_left)]
    #             bound_right = [cos(theta_ort_right), sin(theta_ort_right)]
    #
    #             RVO_BA = [transl_vB_vA, bound_left, bound_right, dist_BA, rad + ROB_RAD]
    #             RVO_BA_all.append(RVO_BA)
    #
    #         # Check if current velocity is in VO region and distance to obstacle
    #         in_vo = False
    #         for RVO_BA in RVO_BA_all:
    #             p_0 = RVO_BA[0]
    #             left = RVO_BA[1]
    #             right = RVO_BA[2]
    #             dif = [vA[0] + pA[0] - p_0[0], vA[1] + pA[1] - p_0[1]]
    #             theta_dif = atan2(dif[1], dif[0])
    #             theta_right = atan2(right[1], right[0])
    #             theta_left = atan2(left[1], left[0])
    #
    #             if in_between(theta_right, theta_dif, theta_left):
    #                 in_vo = True
    #                 if RVO_BA[3] < dist_threshold:  # Check distance to obstacle
    #                     rewards[i] = -1
    #                     break
    #
    #         if not in_vo:
    #             rewards[i] = 0  # No penalty if not in VO region
    #
    #     return rewards

    def VO_reward(self, X, V_current, ws_model, dist_threshold=0.4):
        """
        Compute reward for each robot based on VO region and distance to obstacles.

        Args:
            X: List of robot positions, e.g., [[x1, y1], [x2, y2], ...]
            V_current: List of current velocities, e.g., [[vx1, vy1], [vx2, vy2], ...]
            ws_model: Workspace model containing robot radius and obstacle information.
            dist_threshold: Distance threshold for applying negative reward.

        Returns:
            List of rewards for each robot, e.g., [-1, 0, 0, -1, ...]
        """
        ROB_RAD = ws_model['robot_radius'] - 0.05
        rewards = [0] * len(X)  # Initialize rewards for all robots

        for i in range(len(X)):
            pA = [X[i][0], X[i][1]]  # Current robot position
            vA = [V_current[i][0], V_current[i][1]]  # Current robot velocity
            RVO_BA_all = []

            # Compute RVO regions for other robots
            for j in range(len(X)):
                if i != j:
                    pB = [X[j][0], X[j][1]]
                    vB = [V_current[j][0], V_current[j][1]]

                    # Compute translational velocity and VO bounds
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

            # Compute RVO regions for all circular obstacles
            for obs_idx, obstacle in enumerate(ws_model['circular_obstacles']):
                pB = obstacle['position']  # 障碍物位置
                vB = self.obs_vel[obs_idx]  # 获取对应障碍物的速度
                # print("obs idx=", obs_idx, "VB=", vB)
                radius = obstacle['radius']  # 障碍物半径

                # Compute translational velocity and VO bounds
                transl_vB_vA = [pA[0] + vB[0], pA[1] + vB[1]]
                dist_BA = distance(pA, pB)
                theta_BA = atan2(pB[1] - pA[1], pB[0] - pA[0])

                OVER_APPROX_C2S = 1.3
                rad = radius * OVER_APPROX_C2S  # 使用障碍物的随机半径

                if (rad + ROB_RAD) > dist_BA:
                    dist_BA = rad + ROB_RAD

                theta_BAort = asin((rad + ROB_RAD) / dist_BA)
                theta_ort_left = theta_BA + theta_BAort
                theta_ort_right = theta_BA - theta_BAort
                bound_left = [cos(theta_ort_left), sin(theta_ort_left)]
                bound_right = [cos(theta_ort_right), sin(theta_ort_right)]

                RVO_BA = [transl_vB_vA, bound_left, bound_right, dist_BA, rad + ROB_RAD]
                RVO_BA_all.append(RVO_BA)

            # Check if current velocity is in VO region and distance to obstacle
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
                    if RVO_BA[3] < dist_threshold:  # Check distance to obstacle
                        rewards[i] = -10

            if not in_vo:
                rewards[i] = 0  # No penalty if not in VO region

        return rewards


    # def VO_reward_obs(self, X, V_current, ws_model, dist_threshold=0.4):
    #     """
    #     Compute reward for each robot based on VO region and distance to obstacles.
    #
    #     Args:
    #         X: List of robot positions, e.g., [[x1, y1], [x2, y2], ...]
    #         V_current: List of current velocities, e.g., [[vx1, vy1], [vx2, vy2], ...]
    #         ws_model: Workspace model containing robot radius and obstacle information.
    #         dist_threshold: Distance threshold for applying negative reward.
    #
    #     Returns:
    #         List of rewards for each robot, e.g., [-10, 10, 10, ...]
    #     """
    #     ROB_RAD = ws_model['robot_radius'] + 0.02
    #     rewards = [0] * len(X)  # Initialize rewards for all robots (default: no penalty)
    #     # 限制角度范围为 [-160°, 160°]
    #     ANGLE_LIMIT = 8 * np.pi / 9  # 160° in radians
    #
    #     for i in range(len(X)):
    #         pA = [X[i][0], X[i][1]]  # 当前 UAV 位置
    #         vA = [V_current[i][0], V_current[i][1]]  # 当前 UAV 速度
    #         RVO_BA_all = []
    #
    #         # 计算所有障碍物的 VO 区域
    #         for obs_idx, obstacle in enumerate(ws_model['circular_obstacles']):
    #             pB = obstacle['position']  # 障碍物位置
    #             vB = self.obs_vel[obs_idx]  # 获取对应障碍物的速度
    #             radius = obstacle['radius']  # 障碍物半径
    #             relative_velocity = [vA[0] - vB[0], vA[1] - vB[1]]
    #
    #             # 计算 VO 区域边界
    #             transl_vB_vA = [pA[0] + vB[0], pA[1] + vB[1]]
    #             dist_BA = distance(pA, pB)
    #             theta_BA = atan2(pB[1] - pA[1], pB[0] - pA[0])
    #
    #             OVER_APPROX_C2S = 1.3
    #             rad = radius * OVER_APPROX_C2S  # 使用障碍物的随机半径
    #
    #             if (rad + ROB_RAD) > dist_BA:
    #                 dist_BA = rad + ROB_RAD
    #
    #             theta_BAort = asin((rad + ROB_RAD) / dist_BA)
    #             theta_ort_left = theta_BA + theta_BAort
    #             theta_ort_right = theta_BA - theta_BAort
    #             # 限制角度范围
    #             # theta_ort_left = np.clip(theta_ort_left, -ANGLE_LIMIT, ANGLE_LIMIT)
    #             # theta_ort_right = np.clip(theta_ort_right, -ANGLE_LIMIT, ANGLE_LIMIT)
    #
    #             bound_left = [cos(theta_ort_left), sin(theta_ort_left)]
    #             bound_right = [cos(theta_ort_right), sin(theta_ort_right)]
    #
    #             RVO_BA = [transl_vB_vA, bound_left, bound_right, dist_BA, rad + ROB_RAD]
    #             RVO_BA_all.append(RVO_BA)
    #
    #         # 检查当前速度是否在障碍物的 VO 区域内
    #         in_vo = False
    #         for RVO_BA in RVO_BA_all:
    #             p_0 = RVO_BA[0]
    #             left = RVO_BA[1]
    #             right = RVO_BA[2]
    #             # dif = [vA[0] + pA[0] - p_0[0], vA[1] + pA[1] - p_0[1]]
    #             dif = [relative_velocity[0] + pA[0] - p_0[0], relative_velocity[1] + pA[1] - p_0[1]] # vb - va
    #             theta_dif = atan2(dif[1], dif[0])
    #             theta_right = atan2(right[1], right[0])
    #             theta_left = atan2(left[1], left[0])
    #
    #             if in_between(theta_right, theta_dif, theta_left):
    #                 in_vo = True
    #                 # if RVO_BA[3] < dist_threshold:  # 距离阈值检查
    #                 rewards[i] = -5  # 进入 VO 且距离过近，惩罚
    #                 break
    #
    #         if not in_vo:
    #             rewards[i] = 0  # 未进入 VO 区域，奖励
    #
    #     return rewards

    # def VO_reward_obs(self, X, V_current, ws_model, dist_threshold):
    #     """
    #     Compute reward for each robot based on VO region and distance to obstacles.
    #     The closer the velocity direction is to the VO center, the greater the penalty.
    #
    #     Args:
    #         X: List of robot positions, e.g., [[x1, y1], [x2, y2], ...]
    #         V_current: List of current velocities, e.g., [[vx1, vy1], [vx2, vy2], ...]
    #         ws_model: Workspace model containing robot radius and obstacle information.
    #         dist_threshold: Distance threshold for applying negative reward.
    #
    #     Returns:
    #         List of rewards for each robot, e.g., [-10, 10, 10, ...]
    #     """
    #     ROB_RAD = ws_model['robot_radius'] + 0.02
    #     rewards = [0] * len(X)  # Initialize rewards for all robots (default: no penalty)
    #
    #     for i in range(len(X)):
    #         pA = [X[i][0], X[i][1]]  # 当前 UAV 位置
    #         vA = [V_current[i][0], V_current[i][1]]  # 当前 UAV 速度
    #         RVO_BA_all = []
    #
    #         # 计算所有障碍物的 VO 区域
    #         for obs_idx, obstacle in enumerate(ws_model['circular_obstacles']):
    #             pB = obstacle['position']  # 障碍物位置
    #             vB = self.obs_vel[obs_idx]  # 获取对应障碍物的速度
    #             radius = obstacle['radius']  # 障碍物半径
    #
    #             # 计算 VO 区域边界
    #             transl_vB_vA = [pA[0] + vB[0], pA[1] + vB[1]]
    #             dist_BA = distance(pA, pB)
    #             theta_BA = atan2(pB[1] - pA[1], pB[0] - pA[0])
    #
    #             OVER_APPROX_C2S = 1.3
    #             rad = radius * OVER_APPROX_C2S  # 使用障碍物的随机半径
    #
    #             if (rad + ROB_RAD) > dist_BA:
    #                 dist_BA = rad + ROB_RAD
    #
    #             theta_BAort = asin((rad + ROB_RAD) / dist_BA)
    #             theta_ort_left = theta_BA + theta_BAort
    #             theta_ort_right = theta_BA - theta_BAort
    #
    #             bound_left = [cos(theta_ort_left), sin(theta_ort_left)]
    #             bound_right = [cos(theta_ort_right), sin(theta_ort_right)]
    #
    #             RVO_BA = [transl_vB_vA, bound_left, bound_right, dist_BA, rad + ROB_RAD]
    #             RVO_BA_all.append(RVO_BA)
    #
    #         # 检查当前速度是否在障碍物的 VO 区域内
    #         for RVO_BA in RVO_BA_all:
    #             p_0 = RVO_BA[0]  # VO 中心点
    #             left = RVO_BA[1]  # VO 左边界方向
    #             right = RVO_BA[2]  # VO 右边界方向
    #             dif = [vA[0] + pA[0] - p_0[0], vA[1] + pA[1] - p_0[1]]  # 当前速度方向
    #             theta_dif = atan2(dif[1], dif[0])  # 当前速度方向角
    #             # relative_velocity = [vA[0] - vB[0], vA[1] - vB[1]]
    #             # theta_dif = atan2(relative_velocity[1], relative_velocity[0])  # 相对速度方向角
    #             theta_right = atan2(right[1], right[0])  # VO 右边界方向角
    #             theta_left = atan2(left[1], left[0])  # VO 左边界方向角
    #
    #             # 计算 VO 区域中心方向角
    #             theta_center = (theta_left + theta_right) / 2
    #             if theta_left > theta_right:  # 修正角度跨越 -π 到 π 的情况
    #                 theta_center = (theta_left + theta_right + 2 * np.pi) / 2
    #                 if theta_center > np.pi:
    #                     theta_center -= 2 * np.pi
    #
    #             # 计算速度方向与 VO 中心的夹角
    #             angle_diff = abs(theta_dif - theta_center)
    #             if angle_diff > np.pi:  # 修正夹角范围为 [0, π]
    #                 angle_diff = 2 * np.pi - angle_diff
    #
    #             # 根据夹角大小设计奖励：夹角越小，惩罚越大
    #             if in_between(theta_right, theta_dif, theta_left):
    #                 # 距离越近，惩罚越大；夹角越小，惩罚越大
    #                 dist_penalty = max(0, dist_threshold - RVO_BA[3]) / dist_threshold
    #                 # angle_penalty = 1 - angle_diff / (np.pi / 2)  # 归一化到 [0, 1]
    #                 rewards[i] -= 5 * (dist_penalty)  # 惩罚比例 10
    #             else:
    #                 rewards[i] += 0  # 未进入 VO 区域，给予奖励 1
    #
    #     return rewards


    def VO_reward_obs(self, X, V_current, ws_model, dist_threshold=0.4):
        """
        Compute reward for each robot based on VO region and distance to obstacles.
        The closer the velocity direction is to the VO center, the greater the penalty.

        Args:
            X: List of robot positions, e.g., [[x1, y1], [x2, y2], ...]
            V_current: List of current velocities, e.g., [[vx1, vy1], [vx2, vy2], ...]
            ws_model: Workspace model containing robot radius and obstacle information.
            dist_threshold: Distance threshold for applying negative reward.

        Returns:
            List of rewards for each robot, e.g., [-10, 10, 10, ...]
        """
        ROB_RAD = ws_model['robot_radius'] + 0.02
        rewards = [0] * len(X)  # Initialize rewards for all robots (default: no penalty)

        for i in range(len(X)):
            pA = [X[i][0], X[i][1]]  # 当前 UAV 位置
            vA = [V_current[i][0], V_current[i][1]]  # 当前 UAV 速度
            RVO_BA_all = []

            # 计算所有障碍物的 VO 区域
            for obs_idx, obstacle in enumerate(ws_model['circular_obstacles']):
                pB = obstacle['position']  # 障碍物位置
                vB = self.obs_vel[obs_idx]  # 获取对应障碍物的速度
                radius = obstacle['radius']  # 障碍物半径

                # 计算 VO 区域边界
                transl_vB_vA = [pA[0] + vB[0], pA[1] + vB[1]]
                dist_BA = distance(pA, pB)
                theta_BA = atan2(pB[1] - pA[1], pB[0] - pA[0])

                OVER_APPROX_C2S = 1.3
                rad = radius * OVER_APPROX_C2S  # 使用障碍物的随机半径

                if (rad + ROB_RAD) > dist_BA:
                    dist_BA = rad + ROB_RAD

                theta_BAort = asin((rad + ROB_RAD) / dist_BA)
                theta_ort_left = theta_BA + theta_BAort
                theta_ort_right = theta_BA - theta_BAort

                bound_left = [cos(theta_ort_left), sin(theta_ort_left)]
                bound_right = [cos(theta_ort_right), sin(theta_ort_right)]

                RVO_BA = [transl_vB_vA, bound_left, bound_right, dist_BA, rad + ROB_RAD]
                RVO_BA_all.append(RVO_BA)

            # 检查当前速度是否在障碍物的 VO 区域内
            for RVO_BA in RVO_BA_all:
                p_0 = RVO_BA[0]  # VO 中心点
                left = RVO_BA[1]  # VO 左边界方向
                right = RVO_BA[2]  # VO 右边界方向
                dif = [vA[0] + pA[0] - p_0[0], vA[1] + pA[1] - p_0[1]]  # 当前速度方向
                theta_dif = atan2(dif[1], dif[0])  # 当前速度方向角
                theta_right = atan2(right[1], right[0])  # VO 右边界方向角
                theta_left = atan2(left[1], left[0])  # VO 左边界方向角

                # 计算 VO 区域中心方向角
                theta_center = (theta_left + theta_right) / 2
                if theta_left > theta_right:  # 修正角度跨越 -π 到 π 的情况
                    theta_center = (theta_left + theta_right + 2 * np.pi) / 2
                    if theta_center > np.pi:
                        theta_center -= 2 * np.pi

                # 计算速度方向与 VO 中心的夹角
                angle_diff = abs(theta_dif - theta_center)
                # print(angle_diff)
                if angle_diff > np.pi:  # 修正夹角范围为 [0, π]
                    angle_diff = 2 * np.pi - angle_diff

                # 根据夹角大小设计奖励：夹角越小，惩罚越大
                if in_between(theta_right, theta_dif, theta_left):
                    # 距离越近，惩罚越大；夹角越小，惩罚越大
                    # dist_penalty = max(0, dist_threshold - RVO_BA[3]) / dist_threshold
                    angle_penalty = angle_diff / (np.pi)  # 归一化到 [0, 1]
                    # print(angle_penalty)
                    rewards[i] -= 10 * (angle_penalty)  # 惩罚比例
                else:
                    rewards[i] += 0  # 未进入 VO 区域，给予奖励

        return rewards
    def calculate_expected_collision_time(self, pA, vA, pB, vB, combined_radius):
        """
        Calculate the expected collision time between a robot and an obstacle.

        Args:
            pA: Position of the robot, e.g., [xA, yA].
            vA: Velocity of the robot, e.g., [vxA, vyA].
            pB: Position of the obstacle, e.g., [xB, yB].
            vB: Velocity of the obstacle, e.g., [vxB, vyB].
            combined_radius: Combined radius of the robot and obstacle (robot_radius + obstacle_radius).

        Returns:
            Expected collision time (float). If no collision is expected, return a large value (e.g., np.inf).
        """
        # Calculate relative position and velocity
        delta_p = np.array(pB) - np.array(pA)  # Relative position
        delta_v = np.array(vB) - np.array(vA)  # Relative velocity

        # Quadratic coefficients
        a = np.dot(delta_v, delta_v)  # |delta_v|^2
        b = 2 * np.dot(delta_p, delta_v)  # 2 * (delta_p · delta_v)
        c = np.dot(delta_p, delta_p) - combined_radius ** 2  # |delta_p|^2 - combined_radius^2

        # Discriminant
        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:
            # No real roots, no collision expected
            return np.inf

        # Calculate roots
        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)

        # Choose the smallest positive root
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
        elif t1 > 0:
            return t1
        elif t2 > 0:
            return t2
        else:
            # Both roots are negative, collision already occurred or won't occur
            return np.inf
    def VO_reward_obs_rvo(self, X, V_current, ws_model, dist_threshold=0.4):
        """
        Compute reward for each robot based on VO region and expected collision time.
        Args:
            X: List of robot positions, e.g., [[x1, y1], [x2, y2], ...]
            V_current: List of current velocities, e.g., [[vx1, vy1], [vx2, vy2], ...]
            ws_model: Workspace model containing robot radius and obstacle information.
            dist_threshold: Distance threshold for applying negative reward.

        Returns:
            List of rewards for each robot, e.g., [-10, 10, 10, ...]
        """
        ROB_RAD = ws_model['robot_radius'] + 0.02
        rewards = [0] * len(X)

        # Constants for reward calculation
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
                # print(expected_collision_time)
                if in_between(theta_right, theta_dif, theta_left):
                    if expected_collision_time > 0.1:
                        rewards[i] += c - d * (expected_collision_time + f) ** -1
                    else:
                        rewards[i] -= e * (expected_collision_time + f) ** -1
                else:
                    if expected_collision_time > 4:
                        rewards[i] = a-b*np.linalg.norm(V_current[i] - [0.1, 0.1])

        return rewards



    def speed_reward(self):
        """
        计算每个无人机的速度奖励。
        目标是鼓励无人机速度接近目标速度范围，同时避免危险速度。
        """
        rewards = []
        for i in range(self.num_agents - 1):  # 不包括目标无人机
            vel = self.multi_current_vel[i]
            speed = np.linalg.norm(vel)  # 计算速度的大小

            # 目标速度范围（可以根据任务需求调整）
            target_speed = self.v_max * 0.8  # 目标速度为最大速度的 80%
            tolerance = 0.02  # 允许的速度误差范围

            # 奖励函数：速度接近目标速度时奖励更高，过高或过低的速度会受到惩罚
            if abs(speed - target_speed) <= tolerance:
                reward = 1.0  # 完全匹配目标速度的奖励
            elif speed < target_speed:
                reward = -0.5 * (target_speed - speed)  # 速度过低的惩罚
            else:
                reward = -0.5 * (speed - target_speed)  # 速度过高的惩罚

            rewards.append(reward)
        return rewards

    def cal_rewards_dones(self,IsCollied,last_d,last_pos):
        dones = [False] * self.num_agents
        rewards = np.zeros(self.num_agents)
        mu1 = 0.9 # r_near 0.9
        mu2 = 0.2# r_safe  0.4
        mu3 = 0.0# r_multi_stage 第一阶段0.0，第二阶段0.4
        mu4 = 10 # r_finish 10
        mu5 = 0.1 # 避障 0.2
        d_capture = 0.3
        d_limit = 0.55 # 0.75
        # 获取当前无人机位置
        current_positions = np.array(self.multi_current_pos[:self.num_agents - 1])
        target_position = self.multi_current_pos[-1]
        ## 1 reward for single rounding-up-UAVs:
        for i in range(3):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            pos_target = self.multi_current_pos[-1]
            v_i = np.linalg.norm(vel)
            dire_vec = pos_target - pos
            d = np.linalg.norm(dire_vec) # distance to target

            cos_v_d = np.dot(vel,dire_vec)/(v_i*d + 1e-3)
            r_near = abs(2*v_i/self.v_max)*cos_v_d
            # r_near = min(abs(v_i/self.v_max)*1.0/(d + 1e-5),10)/5
            # print('add 1', mu1 * r_near)
            rewards[i] += mu1 * r_near # TODO: if not get nearer then receive negative reward
            # rewards[i] -= 0.15 * d # 0516里作为测试加入

        ## 2 collision reward for all UAVs:
        for i in range(self.num_agents):
            if IsCollied[i]:
                r_safe = -10
            else:
                lasers = self.multi_current_lasers[i]
                r_safe = (min(lasers) - self.L_sensor - 0.1)/self.L_sensor

            # print('add 2', mu2 * r_safe)
            rewards[i] += mu2 * r_safe
        ws_model = self.get_ws_model()
        vo_rewards = self.VO_reward(self.multi_current_pos, self.multi_current_vel, ws_model,
                                        dist_threshold=0.7) # 0.5
        vo_rewards_float = [float(x) for x in vo_rewards[0:3]]
        # print("VO rewards =", vo_rewards)
        rewards[0:3] += mu5 * np.array(vo_rewards_float)
        # print("rewards =", rewards)
        # for i in range(self.num_agents - 1):
        #
        #     last_distance = np.linalg.norm(last_p[i] - target_position)
        #     current_distance = np.linalg.norm(self.multi_current_pos[i] - target_position)
        #     r_a[i] += mu3 * 10 * (last_distance - current_distance)

        # print(" with VO rewards =", rewards)

        ## 避障 TEST2
        # for i in range(self.num_agents-1):
        #     if IsCollied[i]:
        #         r_vo = vo_rewards[i]
        #     else:
        #         r_vo = 0
        #     rewards[i] += r_vo
        # 速度奖励 弃用
        # for i in range(self.num_agents-1):
        #     if IsCollied[i]:
        #         target_speed = 0  # 目标速度为最大速度的 80%
        #     else:
        #         target_speed = self.v_max * 0.8  # 目标速度为最大速度的 80%
        #     vel = self.multi_current_vel[i]
        #     speed = np.linalg.norm(vel)  # 计算速度的大小
        #
        #
        #
        #     tolerance = 0.02  # 允许的速度误差范围
        #
        #         # 奖励函数：速度接近目标速度时奖励更高，过高或过低的速度会受到惩罚
        #     if abs(speed - target_speed) <= tolerance:
        #         speed_reward = 1.0  # 完全匹配目标速度的奖励
        #     elif speed < target_speed:
        #         speed_reward = -0.5 * (target_speed - speed)  # 速度过低的惩罚
        #     else:
        #         speed_reward = -0.5 * (speed - target_speed)  # 速度过高的惩
        #     rewards[i] += speed_reward

        # 接近目标奖励
        # for i in range(self.num_agents - 1):
        #
        #     last_distance = np.linalg.norm(last_pos[i] - target_position)
        #     # print(f"agt last dis {i}", last_distance)
        #     current_distance = np.linalg.norm(self.multi_current_pos[i] - target_position)
        #
        #     rewards[i] += 0.1*(last_distance - current_distance)
        #     print(f"agt {i}", rewards[i])
        #     # print(f"agt current dis {i}", 100*(last_distance - current_distance))



        ## 3 multi-stage's reward for rounding-up-UAVs
        p0 = self.multi_current_pos[0]
        p1 = self.multi_current_pos[1]
        p2 = self.multi_current_pos[2]
        pe = self.multi_current_pos[-1]
        S1 = cal_triangle_S(p0,p1,pe)
        S2 = cal_triangle_S(p1,p2,pe)
        S3 = cal_triangle_S(p2,p0,pe)
        S4 = cal_triangle_S(p0,p1,p2)
        d1 = np.linalg.norm(p0-pe)
        d2 = np.linalg.norm(p1-pe)
        d3 = np.linalg.norm(p2-pe)
        Sum_S = S1 + S2 + S3
        Sum_d = d1 + d2 + d3
        Sum_last_d = sum(last_d)
        # 3.1 reward for target UAV:
        rewards[-1] += np.clip(2 * (Sum_d - Sum_last_d),-2,2)
        # print(rewards[-1])
        # 3.2 stage-1 track

        # if Sum_d >= d_limit and all(d >= d_capture for d in [d1, d2, d3]):

        if Sum_S > S4 and Sum_d >= d_limit and all(d >= d_capture for d in [d1, d2, d3]):
            r_track = - Sum_d/max([d1,d2,d3])
            rewards[0:3] += mu3*r_track
        # 3.3 stage-2 encircle
        elif Sum_S > S4 and (Sum_d < d_limit or any(d >= d_capture for d in [d1, d2, d3])):
            r_encircle = -1/3*np.log(Sum_S - S4 + 1)
            rewards[0:3] += mu3*r_encircle
        # 3.4 stage-3 capture
        elif Sum_S == S4 and any(d > d_capture for d in [d1,d2,d3]):
            r_capture = np.exp((Sum_last_d - Sum_d)/(3*self.v_max))
            rewards[0:3] += mu3*r_capture

        ## 4 finish rewards
        if Sum_S == S4 and all(d <= d_capture for d in [d1,d2,d3]):
            print('add 4', mu4*10)
            rewards[0:3] += mu4*10
            dones = [True] * self.num_agents
        return rewards,dones

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
                _current_lasers, done = update_lasers(pos,obs_pos,r,self.L_sensor,self.num_lasers,self.length)
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

        # load UAV icon
        uav_icon = mpimg.imread('UAV.png')
        # icon_height, icon_width, _ = uav_icon.shape

        # plot round-up-UAVs
        for i in range(self.num_agents - 1):
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            self.history_positions[i].append(pos)
            trajectory = np.array(self.history_positions[i])
            # plot trajectory
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.3)
            # Calculate the angle of the velocity vector
            angle = np.arctan2(vel[1], vel[0])

            # plt.scatter(pos[0], pos[1], c='b', label='hunter')
            t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            # plt.imshow(uav_icon, extent=(pos[0] - 0.05, pos[0] + 0.05, pos[1] - 0.05, pos[1] + 0.05))
            # plt.imshow(uav_icon, transform=t + plt.gca().transData, extent=(pos[0] - 0.05, pos[0] + 0.05, pos[1] - 0.05, pos[1] + 0.05))
            icon_size = 0.1  # Adjust this size to your icon's aspect ratio
            plt.imshow(uav_icon, transform=t + plt.gca().transData, extent=(-icon_size/2, icon_size/2, -icon_size/2, icon_size/2))

            # # Visualize laser rays for each UAV(can be closed when unneeded)
            # lasers = self.multi_current_lasers[i]
            # angles = np.linspace(0, 2 * np.pi, len(lasers), endpoint=False)

            # for angle, laser_length in zip(angles, lasers):
            #     laser_end = np.array(pos) + np.array([laser_length * np.cos(angle), laser_length * np.sin(angle)])
            #     plt.plot([pos[0], laser_end[0]], [pos[1], laser_end[1]], 'b-', alpha=0.2)

        # plot target
        plt.scatter(self.multi_current_pos[-1][0], self.multi_current_pos[-1][1], c='r', label='Target')
        self.history_positions[-1].append(copy.deepcopy(self.multi_current_pos[-1]))
        trajectory = np.array(self.history_positions[-1])
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.3)

        for obstacle in self.obstacles:
            circle = plt.Circle(obstacle.position, obstacle.radius, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)
        plt.xlim(-0.1, self.length+0.1)
        plt.ylim(-0.1, self.length+0.1)
        plt.draw()
        plt.legend()
        # plt.pause(0.01)
        # Save the current figure to a buffer
        canvas = agg.FigureCanvasAgg(plt.gcf())
        canvas.draw()
        buf = canvas.buffer_rgba()

        # Convert buffer to a NumPy array
        image = np.asarray(buf)
        return image

    def render_anime_vo(self, frame_num):
        plt.clf()

        # 加载 UAV 图片
        uav_icon = mpimg.imread('UAV.png')

        # 绘制每个 UAV 的轨迹和当前位置
        for i in range(self.num_agents - 1):
            # 获取 UAV 当前位置和速度
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            angle = np.arctan2(vel[1], vel[0])  # 计算旋转角度
            self.history_positions[i].append(pos)  # 保存历史轨迹

            # 绘制轨迹
            trajectory = np.array(self.history_positions[i])
            for j in range(len(trajectory) - 1):
                color = cm.viridis(j / len(trajectory))  # 使用 viridis colormap
                plt.plot(trajectory[j:j + 2, 0], trajectory[j:j + 2, 1], color=color, alpha=0.7)

            # 绘制 UAV 图片
            t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            icon_size = 0.1  # 调整图片大小
            plt.imshow(uav_icon, transform=t + plt.gca().transData,
                       extent=(-icon_size / 2, icon_size / 2, -icon_size / 2, icon_size / 2))

        # 绘制目标点（最后一个 agent 的位置）
        plt.scatter(self.multi_current_pos[-1][0], self.multi_current_pos[-1][1], c='r', label='Target')
        pos_e = copy.deepcopy(self.multi_current_pos[-1])
        self.history_positions[-1].append(pos_e)
        trajectory = np.array(self.history_positions[-1])
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.3)

        # 绘制障碍物
        for obstacle in self.obstacles:
            circle = plt.Circle(obstacle.position, obstacle.radius, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)

        # 绘制 VO 区域
        ws_model = self.get_ws_model()  # 获取工作空间模型
        self.VO_Plot(self.multi_current_pos, self.multi_current_vel, ws_model, FLAG=True, ax=plt.gca())

        # 设置图像范围
        plt.xlim(-0.1, self.length + 0.1)
        plt.ylim(-0.1, self.length + 0.1)

        # 绘制图像
        plt.draw()

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
                color = cm.viridis(j / len(trajectory))  # 使用 viridis colormap
                plt.plot(trajectory[j:j+2, 0], trajectory[j:j+2, 1], color=color, alpha=0.7)
            # plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=1)

            t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            icon_size = 0.1
            plt.imshow(uav_icon, transform=t + plt.gca().transData, extent=(-icon_size/2, icon_size/2, -icon_size/2, icon_size/2))

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
            self.position = np.random.uniform(low=0.45, high=length-0.55, size=(2,))
            angle = np.random.uniform(0, 2 * np.pi)
            speed = 0.025
            self.velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])
            self.radius = np.random.uniform(0.14, 0.18)
        elif mode == "fixed":
            # 固定轨迹模式
            position_offset = np.random.uniform(-0.5, 0.5, size=2)  # 随机位置偏移
            velocity_offset = np.random.uniform(-0.04, 0.04, size=2)  # 随机速度偏移

            if index % 2 == 0:
                # 从 (0, 2) 出发
                start_x = 0
                start_y = 2 - (index // 2) * (2 / (total_obstacles // 2))
                self.position = np.array([start_x, start_y]) + position_offset  # 加入随机位置偏移
                if index == 1:
                    self.velocity = np.array([0.03, -0.02]) + velocity_offset  # 加入随机速度偏移
                else:
                    self.velocity = np.array([0.02, -0.03]) + velocity_offset  # 加入随机速度偏移
            else:
                # 从 (2, 0) 出发
                start_x = 2
                start_y = (index // 2) * (2 / (total_obstacles // 2))
                self.position = np.array([start_x, start_y]) + position_offset  # 加入随机位置偏移
                self.velocity = np.array([-0.02, 0.03]) + velocity_offset  # 加入随机速度偏移

            self.radius = np.random.uniform(0.15, 0.20)
