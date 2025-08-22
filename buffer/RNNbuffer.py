import numpy as np
import torch

class RNNMultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims, 
            n_actions, n_agents, batch_size, hidden_state_dim):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)

        # 初始化隐藏状态存储
        # 初始化隐藏状态存储为三维数组：存储容量 x 代理数量 x 隐藏状态维度
        self.hidden_memory = np.zeros((self.n_agents, self.mem_size,4, hidden_state_dim))
        self.cell_memory = np.zeros((self.n_agents, self.mem_size, 4, hidden_state_dim))
        self.next_hidden_memory = np.zeros((self.n_agents,self.mem_size,4,  hidden_state_dim))
        self.next_cell_memory = np.zeros(( self.n_agents,self.mem_size, 4,hidden_state_dim))

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
        h_0 = torch.zeros((batch_size, num_layers, hidden_size), dtype=torch.float).to(device)
        c_0 = torch.zeros((batch_size, num_layers, hidden_size), dtype=torch.float).to(device)
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
            h, c = hidden_states[agent_idx]
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

        self.mem_cntr += 1

    def sample_buffer(self):
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

    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True
