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


class RNNPERMultiAgentReplayBuffer:
    """
    带有优先经验回放的多智能体经验缓冲区
    """
    def __init__(self, max_size, critic_dims, actor_dims,
                 n_actions, n_agents, batch_size, hidden_state_dim, alpha=0.6, beta=0.4, abs_err_upper=15, lstm_num_layers =8):
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

        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)
        self.lstm_num_layers = lstm_num_layers
        # 初始化隐藏状态存储
        # 初始化隐藏状态存储为三维数组：存储容量 x 代理数量 x 隐藏状态维度
        self.hidden_memory = np.zeros((self.n_agents, self.mem_size,self.lstm_num_layers, hidden_state_dim))
        self.cell_memory = np.zeros((self.n_agents, self.mem_size, self.lstm_num_layers, hidden_state_dim))
        self.next_hidden_memory = np.zeros((self.n_agents,self.mem_size,self.lstm_num_layers,  hidden_state_dim))
        self.next_cell_memory = np.zeros(( self.n_agents,self.mem_size, self.lstm_num_layers,hidden_state_dim))
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []

        for i in range(self.n_agents):
            self.actor_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_new_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action_memory.append(
                            np.zeros((self.mem_size, self.n_actions)))
    def init_hidden_memory(self, batch_size, num_layers,hidden_size, device):
        """
        初始化隐藏状态和细胞状态，用于递归网络（如 LSTM）。
        h0=(num_layers, batch_size, hidden_size)
        c0=(num_layers, batch_size, hidden_size)
        :param num_layers: LSTM 的层数
        :param hidden_size: LSTM 隐藏状态的维度
        :param device: 设备（'cpu' 或 'cuda'）
        :return: (h_0, c_0) 初始隐藏状态和细胞状态
        """
        h_0 = T.zeros((batch_size, num_layers, hidden_size), dtype=T.float).to(device)
        c_0 = T.zeros((batch_size, num_layers, hidden_size), dtype=T.float).to(device)
        return h_0, c_0

    def store_transition(self, raw_obs, state, action, reward,
                               raw_obs_, state_, done, hidden_states, next_hidden_states):
        # this introduces a bug: if we fill up the memory capacity and then
        # zero out our actor memory, the critic will still have memories to access
        # while the actor will have nothing but zeros to sample. Obviously
        # not what we intend.
        # In reality, there's no problem with just using the same index
        # for both the actor and critic states. I'm not sure why I thought
        # this was necessary in the first place. Sorry for the confusion!

        #if self.mem_cntr % self.mem_size == 0 and self.mem_cntr > 0:
        #    self.init_actor_memory()

        index = self.mem_cntr % self.mem_size

        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[agent_idx]
            # 存储每个代理的隐藏状态
            h, c = hidden_states[agent_idx] # [num_layers, 1 ,batch_size]
            nh, nc = next_hidden_states[agent_idx]

            # Detach 隐藏状态，防止梯度传播到前一个时间步
            self.hidden_memory[agent_idx,index, :,:] = h.detach().cpu().numpy().squeeze(1)
            self.cell_memory[ agent_idx,index, :,:] = c.detach().cpu().numpy().squeeze(1)
            self.next_hidden_memory[agent_idx,index,:,  :] = nh.detach().cpu().numpy().squeeze(1)
            self.next_cell_memory[ agent_idx,index, :,:] = nc.detach().cpu().numpy().squeeze(1)

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        # h, c = hidden_states
        # nh, nc = next_hidden_states
        #
        # # Detach 隐藏状态，防止梯度传播到前一个时间步
        # # Detach 隐藏状态，防止梯度传播到前一个时间步
        # self.hidden_memory[index] = h.detach().cpu().numpy().reshape(-1).tolist()# [1000000,256]
        # self.cell_memory[index] = c.detach().cpu().numpy().reshape(-1).tolist()# [1000000,256]
        # self.next_hidden_memory[index] = nh.detach().cpu().numpy().reshape(-1).tolist()# [1000000,256]
        # self.next_cell_memory[index] = nc.detach().cpu().numpy().reshape(-1).tolist()# [1000000,256]

        # 为新经验分配一个较大的初始优先级
        max_p = np.max(self.sum_tree.tree[-self.mem_size:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.sum_tree.add(max_p, index)
        self.mem_cntr += 1

    def sample_buffer(self, critic_networks, gamma, hidden_states):
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
        hidden = []
        next_hidden = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])
            h = self.hidden_memory[agent_idx, batch, :]  # 原始二维数据 [batch_size, hidden_dim]
            h = h.reshape(-1, self.batch_size, h.shape[-1])  # 变换为三维 [batch_size, seq_len, hidden_dim]

            c = self.cell_memory[agent_idx, batch, :]
            c = c.reshape(-1, self.batch_size, c.shape[-1])

            nh = self.next_hidden_memory[agent_idx, batch, :]
            nh = nh.reshape(-1, self.batch_size, nh.shape[-1])

            nc = self.next_cell_memory[agent_idx, batch, :]
            nc = nc.reshape(-1, self.batch_size, nc.shape[-1])

            # 构造隐藏状态元组
            hidden.append((h, c))
            next_hidden.append((nh, nc))

        # 使用每个智能体的 critic_network 计算 TD误差
        states_tensor = T.tensor(states, dtype=T.float32).to(self.device)
        states_tensor_ = T.tensor(states_, dtype=T.float32).to(self.device)
        rewards_tensor = T.tensor(rewards, dtype=T.float32).to(self.device)
        terminal_tensor = T.tensor(terminal, dtype=T.float32).to(self.device)
        actions_tensor = T.tensor(np.concatenate(actions, axis=1), dtype=T.float32).to(self.device)
        # new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)

        all_agents_new_actions = []
        old_agents_actions = []
        td_errors = []
        with T.no_grad():
            for agent_idx, critic_network in enumerate(critic_networks):
                (h, c) = hidden_states[agent_idx]  # 解压当前隐藏状态 (h, c)
                h = T.tensor(h, dtype=T.float).to(self.device)
                c = T.tensor(c, dtype=T.float).to(self.device)

                saved_hidden_states = (h, c)
                # 计算下一状态的 Q 值
                q_next = critic_network(states_tensor_, actions_tensor,saved_hidden_states).flatten()
                # 计算目标 Q 值
                q_target = rewards_tensor[:, agent_idx] + gamma * (1 - terminal_tensor[:, 0].int()) * q_next
                # 计算当前 Q 值
                q_current = critic_network(states_tensor, actions_tensor, saved_hidden_states).flatten()
                # 计算 TD 误差
                td_error = q_target - q_current
                td_errors.append(td_error.cpu().numpy())  # 转换为 numpy 数组
        # 更新优先级
        max_td_error = np.max(np.abs(td_errors), axis=0)  # 执行最大值操作
        # print(max_td_error.shape)
        self.update_priorities(indices, max_td_error)
        return actor_states, states, actions, rewards, \
            actor_new_states, states_, terminal, indices, ISWeights, hidden, next_hidden

    def sample_buffer_init(self):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        actor_states = []
        actor_new_states = []
        actions = []
        hidden = []
        next_hidden = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])
            h = self.hidden_memory[agent_idx, batch, :]  # 原始二维数据 [batch_size, hidden_dim]
            h = h.reshape(-1, self.batch_size, h.shape[-1])  # 变换为三维 [batch_size, seq_len, hidden_dim]

            c = self.cell_memory[agent_idx, batch, :]
            c = c.reshape(-1, self.batch_size, c.shape[-1])

            nh = self.next_hidden_memory[agent_idx, batch, :]
            nh = nh.reshape(-1, self.batch_size, nh.shape[-1])

            nc = self.next_cell_memory[agent_idx, batch, :]
            nc = nc.reshape(-1, self.batch_size, nc.shape[-1])

            # 构造隐藏状态元组
            hidden.append((h, c))
            next_hidden.append((nh, nc))
        # 采样隐藏状态 batch 个
        # h = [self.hidden_memory[idx] for idx in batch]
        # c = [self.cell_memory[idx] for idx in batch]
        # nh = [self.next_hidden_memory[idx] for idx in batch]
        # nc = [self.next_cell_memory[idx] for idx in batch]

        # 构造隐藏状态元组
        # hidden = (h, c)
        # next_hidden = (nh, nc)

        return actor_states, states, actions, rewards, \
            actor_new_states, states_, terminal, hidden, next_hidden


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