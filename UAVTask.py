import numpy as np


class UAVTaskEvaluator:
    def __init__(self, target_position, max_time_steps, dones, collision_penalty=50, success_reward=1000,
                 path_weight=50, time_weight=20):
        """
        初始化评测器
        :param target_position: 目标点的位置 (x, y, z)
        :param max_time_steps: 最大允许时间步数
        :param collision_penalty: 每次碰撞的惩罚
        :param success_reward: 成功到达目标的奖励
        :param path_weight: 路径效率的权重
        :param time_weight: 时间效率的权重
        """
        # self.env = env
        self.target_position = np.array(target_position)
        self.max_time_steps = max_time_steps
        self.collision_penalty = collision_penalty
        self.success_reward = success_reward
        self.path_weight = path_weight
        self.time_weight = time_weight
        self.dones = dones
        self.success_count = 0  # 成功次数
        # self.total_rounds = 0  # 总回合数
        # self.success_rate = 0  # 成功率

    def evaluate(self, trajectory, collisions, target_position):
        """
        评测无人机任务完成度
        :param trajectory: 无人机的轨迹，形状为 (time_steps, 2) 或 (time_steps, 3)，每一行是 (x, y) 或 (x, y, z) 坐标
        :param collisions: 碰撞记录，布尔列表，长度为 time_steps
        :return: 评测结果字典
        """
        self.target_position = np.array(target_position)
        trajectory = np.array(trajectory)
        time_steps = len(trajectory)

        # 检查轨迹是否为空
        if time_steps == 0:
            raise ValueError("Trajectory is empty. Cannot evaluate the task.")

        # 确保目标位置和轨迹维度一致（忽略多余维度）

        # 1. 任务完成度：检查是否到达目标点
        # success = np.linalg.norm(trajectory[-1] - self.target_position) < 0.1  # 允许一定误差范围
        # completion_score = self.success_reward if success else 0

        if all(self.dones):
            # print("sueecss info:", self.dones)
            completion_score = self.success_reward
        else:
            completion_score = 0

        # 2. 路径效率：计算路径长度与最短路径的比值
        path_length = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        optimal_path_length = np.linalg.norm(trajectory[0] - self.target_position)
        path_efficiency = optimal_path_length / path_length if path_length > 0 else 0

        # 3. 安全性：统计碰撞次数
        collisions = [item for sublist in collisions for item in sublist]  # 展平嵌套列表
        collision_count = sum(collisions)
        safety_score = -self.collision_penalty * collision_count

        # 4. 时间效率：计算完成任务所用时间步数
        time_efficiency = 1 - (time_steps / self.max_time_steps)
        time_efficiency = max(0, time_efficiency)  # 确保时间效率不为负

        # 综合得分（加权平均）
        total_score = (
            completion_score +
            path_efficiency * self.path_weight +  # 路径效率权重
            safety_score +
            time_efficiency * self.time_weight  # 时间效率权重
        )

        # 返回评测结果
        return {
            "completion_score": completion_score,
            "path_efficiency": path_efficiency,
            "collision_count": collision_count,
            "safety_score": safety_score,
            "time_efficiency": time_efficiency,
            "total_score": total_score
        }

    def success_evaluate(self, done , total_rounds):
        # 判断是否成功（依据：所有无人机到达目标点）
        if all(done):  # 如果 dones 全为 True，则表示成功
            self.success_count += 1
        # print(self.success_count)
        total_rounds += 1
        # 增加总回合数
        # self.total_rounds += 1
        # print(total_rounds)

        # 每隔 100 回合计算成功率
        if total_rounds % 10 == 0:
            success_rate = self.success_count / total_rounds
            return success_rate, self.success_count
        else:
            return None, self.success_count

