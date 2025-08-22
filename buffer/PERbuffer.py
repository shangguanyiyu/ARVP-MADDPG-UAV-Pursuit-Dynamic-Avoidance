import numpy as np
import torch as T


class SumTree:
    """
    SumTree 数据结构，用于存储优先级和支持快速采样。
    """
    def __init__(self, capacity):
        self.capacity = capacity  # SumTree 的容量（叶子节点数量）
        self.tree = np.zeros(2 * capacity - 1)  # 二叉树数组表示
        self.data = [None] * capacity  # 存储实际经验
        self.data_pointer = 0  # 当前存储位置指针

    def add(self, p, data):
        """
        添加新经验及其优先级。
        """
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # 存储经验
        self.update(tree_idx, p)  # 更新优先级

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0  # 循环覆盖最旧的数据

    def update(self, tree_idx, p):
        """
        更新指定叶子节点的优先级，并向上传递变化。
        """
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:  # 向上传递变化
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        根据随机值 v 采样叶子节点。
        """
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1  # 左子节点索引
            cr_idx = cl_idx + 1  # 右子节点索引
            if cl_idx >= len(self.tree):  # 到达叶子节点
                leaf_idx = parent_idx
                break
            else:  # 向下搜索
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        """
        返回所有优先级的总和（根节点值）。
        """
        return self.tree[0]


class PERMultiAgentReplayBuffer:
    """
    带有优先经验回放的多智能体经验缓冲区
    """
    def __init__(self, max_size, critic_dims, actor_dims,
                 n_actions, n_agents, batch_size, alpha=0.6, beta=0.4, abs_err_upper=15):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.alpha = alpha  # 优先级的敏感度
        self.beta = beta  # 用于重要性采样权重
        self.abs_err_upper = abs_err_upper  # 优先级的上限

        # 使用 SumTree 存储优先级和经验
        self.sum_tree = SumTree(max_size)

        # 初始化经验存储
        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)

        self.init_actor_memory()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def init_actor_memory(self):
        self.actor_state_memory = []  # 当前状态记忆
        self.actor_new_state_memory = []  # 下一状态记忆
        self.actor_action_memory = []  # 动作记忆

        for i in range(self.n_agents):
            self.actor_state_memory.append(
                np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_new_state_memory.append(
                np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action_memory.append(
                np.zeros((self.mem_size, self.n_actions)))

    def store_transition(self, raw_obs, state, action, reward,
                         raw_obs_, state_, done):
        """
        存储多智能体的经验
        """
        index = self.mem_cntr % self.mem_size

        # 存储多智能体的状态、动作、奖励等
        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[agent_idx]

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        # 为新经验分配一个较大的初始优先级
        max_p = np.max(self.sum_tree.tree[-self.mem_size:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.sum_tree.add(max_p, index)

        self.mem_cntr += 1

    def sample_buffer(self, critic_networks, gamma):
        """
        根据优先级采样经验，并分别计算每个智能体的 TD 误差。
        :param critic_networks: 每个智能体的 critic 网络（列表）
        :param gamma: 折扣因子
        :return: 采样的经验和每个智能体的 TD 误差
        """
        max_mem = min(self.mem_cntr, self.mem_size)
        segment = self.sum_tree.total_p / self.batch_size  # 总优先级分段
        indices = []
        priorities = []
        ISWeights = np.zeros(self.batch_size, dtype=np.float32)

        for i in range(self.batch_size):
            a, b = segment * i, segment * (i + 1)
            v = np.random.uniform(a, b)
            idx, priority, data_idx = self.sum_tree.get_leaf(v)
            indices.append(data_idx)
            priorities.append(priority)

        # 计算重要性采样权重
        total_p = self.sum_tree.total_p
        min_p = np.min(self.sum_tree.tree[-max_mem:]) / total_p
        for i, p in enumerate(priorities):
            ISWeights[i] = (p / total_p) ** -self.beta
        ISWeights /= ISWeights.max()  # 归一化

        # 提取对应的经验
        batch = np.array(indices)
        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        actor_states = []
        actor_new_states = []
        actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])

        # 使用每个智能体的 critic_network 计算 TD误差
        states_tensor = T.tensor(states, dtype=T.float32).to(self.device)
        states_tensor_ = T.tensor(states_, dtype=T.float32).to(self.device)
        rewards_tensor = T.tensor(rewards, dtype=T.float32).to(self.device)
        terminal_tensor = T.tensor(terminal, dtype=T.float32).to(self.device)
        actions_tensor = T.tensor(np.concatenate(actions, axis=1), dtype=T.float32).to(self.device)

        td_errors = []
        with T.no_grad():
            for agent_idx, critic_network in enumerate(critic_networks):
                # 计算下一状态的 Q 值
                q_next = critic_network(states_tensor_, actions_tensor).flatten()
                # 计算目标 Q 值
                q_target = rewards_tensor[:, agent_idx] + gamma * (1 - terminal_tensor[:, 0].int()) * q_next
                # 计算当前 Q 值
                q_current = critic_network(states_tensor, actions_tensor).flatten()
                # 计算 TD 误差
                td_error = q_target - q_current
                td_errors.append(td_error.cpu().numpy())  # 转换为 numpy 数组
        # 更新优先级
        self.update_priorities(indices, np.max(np.abs(td_errors), axis=0))
        return actor_states, states, actions, rewards, \
            actor_new_states, states_, terminal, indices, ISWeights

    def sample_buffer_withcor(self, uav_position, sigma1, sigma2, sigma3):
        """
        根据合围任务相关性（calculate_fr）采样经验
        :param target_position: 目标位置 (x, y) 或 (x, y, z)
        :param sigma1: calculate_fr 的第一项权重
        :param sigma2: calculate_fr 的第二项权重
        :param sigma3: calculate_fr 的第三项权重
        """
        max_mem = min(self.mem_cntr, self.mem_size)
        segment = self.sum_tree.total_p / self.batch_size  # 总优先级分段
        indices = []
        priorities = []
        ISWeights = np.zeros(self.batch_size, dtype=np.float32)
        target_position = uav_position[-1]

        for i in range(self.batch_size):
            a, b = segment * i, segment * (i + 1)
            v = np.random.uniform(a, b)
            idx, priority, data_idx = self.sum_tree.get_leaf(v)
            indices.append(data_idx)
            priorities.append(priority)

        # 计算 UAV 合围任务的相关性分数
        fr_scores = []
        for idx in indices:
            uav_positions = uav_position[idx]  # 获取存储的 UAV 位置
            fr_score = self.calculate_fr(uav_positions, target_position, sigma1, sigma2, sigma3)
            fr_scores.append(fr_score)

        # 根据相关性分数调整优先级
        total_fr = sum(fr_scores)
        normalized_fr_scores = [score / total_fr for score in fr_scores]

        # 更新 SumTree 的优先级
        for data_idx, fr_score in zip(indices, normalized_fr_scores):# TODO 如何更新SumTree的优先级？
            tree_idx = data_idx + self.sum_tree.capacity - 1  # 将 data_idx 转换为树中的叶子节点索引
            self.sum_tree.update(tree_idx, fr_score)

        # 计算重要性采样权重
        total_p = self.sum_tree.total_p
        min_p = np.min(self.sum_tree.tree[-max_mem:]) / total_p
        for i, p in enumerate(priorities):
            ISWeights[i] = (p / total_p) ** -self.beta
        ISWeights /= ISWeights.max()  # 归一化

        # 提取对应的经验
        batch = np.array(indices)
        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        actor_states = []
        actor_new_states = []
        actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])

        return actor_states, states, actions, rewards, \
            actor_new_states, states_, terminal, indices, ISWeights, terminal

    def update_priorities(self, indices, td_errors):
        """
        根据 TD-误差更新优先级
        """
        for i, idx in enumerate(indices):
            p = np.abs(td_errors[i]) + 1e-5  # 避免优先级为 0
            p = np.clip(p, 0, self.abs_err_upper)
            p = p ** self.alpha
            self.sum_tree.update(idx + self.mem_size - 1, p)

    def ready(self):
        return self.mem_cntr >= self.batch_size

    def calculate_barycenter(self, uav_positions):
        """
        计算 UAV 合围网络的重心
        :param uav_positions: UAV 的位置数组，形状为 (n, 2) 或 (n, 3)
        :return: 重心坐标
        """
        return np.mean(uav_positions, axis=0)

    def calculate_similarity(self,pos1, pos2):
        """
        计算两个位置之间的相似性（这里使用余弦相似度作为示例）
        :param pos1: 第一个位置 (x, y) 或 (x, y, z)
        :param pos2: 第二个位置 (x, y) 或 (x, y, z)
        :return: 相似性分数
        """
        pos1 = np.array(pos1)
        pos2 = np.array(pos2)
        norm1 = np.linalg.norm(pos1)
        norm2 = np.linalg.norm(pos2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return np.dot(pos1, pos2) / (norm1 * norm2)

    def calculate_fr(self,uav_positions, target_position, sigma1, sigma2, sigma3):
        """
        计算相关性函数 fr(s_t)
        :param uav_positions: 所有 UAV 的位置数组，形状为 (n, 2) 或 (n, 3)
        :param target_position: 目标的位置 (x, y) 或 (x, y, z)
        :param sigma1: 第一项的权重
        :param sigma2: 第二项的权重
        :param sigma3: 第三项的权重
        :return: 相关性分数 fr(s_t)
        """
        n = len(uav_positions)
        barycenter = self.calculate_barycenter(uav_positions)

        # 第一项: UAVs 之间的相似性
        term1 = 0
        for i in range(n - 1):
            term1 += self.calculate_similarity(uav_positions[i], uav_positions[i + 1])

        # 第二项: UAVs 和目标的相对相似性
        term2 = 0
        for i in range(n - 1):
            term2 += self.calculate_similarity(uav_positions[i], target_position)

        # 第三项: UAVs 到重心的距离
        term3 = 0
        for i in range(n):
            term3 += np.linalg.norm(uav_positions[i] - barycenter)

        # 加权求和
        fr_st = sigma1 * term1 + sigma2 * term2 + sigma3 * term3
        return fr_st
