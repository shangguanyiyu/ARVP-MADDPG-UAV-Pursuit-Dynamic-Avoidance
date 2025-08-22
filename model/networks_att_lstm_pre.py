import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ATT import Attention_ATT_2


class CriticNetwork_1(nn.Module):
    id_num: int = 0
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims,
                 n_agents, n_actions, name, chkpt_dir, dim_info, agent_id, hidden_dim, lstm_hidden_dim, is_recurrent=True, num_layers=8):
        super(CriticNetwork_1, self).__init__()

        self.recurrent = is_recurrent
        self.chkpt_file = os.path.join(chkpt_dir, name)
        # 智能体编号
        self.id = CriticNetwork_1.id_num
        CriticNetwork_1.id_num += 1
        # print(CriticNetwork.id_num)
        # 检查 dim_info 的键数量是否足够
        self.hidden_dim = hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        if self.id >= len(dim_info):
            raise ValueError(f"agent_id {self.id} is out of range for dim_info with {len(dim_info)} keys.")

        # 动作维度序号 # [1,2,1] -> [0,1,3,4] # 海象运算符:= 既计算某个值，也将其赋给一个变量
        a_d = [dim_info[id][1] for id in dim_info.keys()]
        sum_ = 0
        self.a_d = [0] + [sum_ := sum_ + i for i in a_d]  # self.a_d 和 self.s_d 用于forward中切片
        # 智能体个数
        self.agent_n = len(dim_info)
        agent_id = list(dim_info.keys())[self.id]

        # 注意力相关
        ''' 这里是每个智能体给一个attention模块'''
        self.attention_modules = Attention_ATT_2(hidden_dim, hidden_dim, hidden_dim, 4)
        # 输入准备
        ## 所有智能体的状态和当前智能体动作 维度
        self.encoder_input_dim = sum([dim_info[agent_id_][0] for agent_id_ in dim_info.keys()]) + dim_info[agent_id][1]
        self.encoder_input_fc = T.nn.Linear(self.encoder_input_dim, hidden_dim)
        ## 其余智能体的动作 维度
        self.decoder_input_dim = sum([dim_info[agent_id_][1] for agent_id_ in dim_info.keys() if agent_id_ != agent_id])
        self.decoder_input_fc = T.nn.Linear(self.decoder_input_dim, hidden_dim)

        # q_value 输出
        if self.recurrent:
            self.lstm = T.nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True, num_layers=num_layers)
            self.fc1 = T.nn.Linear(lstm_hidden_dim, hidden_dim)
            self.fc2 = T.nn.Linear(hidden_dim, 1)
        else:
            self.fc1 = T.nn.Linear(hidden_dim, hidden_dim)
            self.fc2 = T.nn.Linear(hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.7)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    # def forward(self, state, action):
    #     x = F.relu(self.fc1(T.cat([state, action], dim=1)))
    #     x = F.relu(self.fc2(x))
    #     q = self.q(x)
    #
    #     return q

    def forward(self, s, a, hidden_state):
        # s: 拼接后的状态张量，形状 [batch_size, state_dim]
        # a: 拼接后的动作张量，形状 [batch_size, action_dim]

        # 当前智能体的动作
        a_i = a[:, self.a_d[self.id]:self.a_d[self.id + 1]]  # self.a_d 用于定位当前智能体的动作切片

        # 编码器输入：全局状态 + 当前智能体动作
        encoder_input = self.encoder_input_fc(T.cat((s, a_i), dim=1))  # [batch_size, state_dim + action_dim]

        # 解码器输入：其他智能体的动作
        decoder_input = self.decoder_input_fc(
            T.cat([a[:, self.a_d[i]:self.a_d[i + 1]] for i in range(self.agent_n) if i != self.id], dim=1)
        )  # [batch_size, action_dim * (agent_count - 1)]

        # 注意力模块
        contextual_vector = self.attention_modules(encoder_input, decoder_input)  # [batch_size, hidden_dim]
        # contextual_vector = contextual_vector + encoder_input
        if self.recurrent:
            contextual_vector = contextual_vector.unsqueeze(1)  # 如果是单步输入，添加时间维度 [256, 1, 128]
            self.lstm.flatten_parameters()
            x, hidden_state = self.lstm(contextual_vector, hidden_state) # x = [batch_size, lstm_hidden_dim]
        # 加入残差连接
        #     x = x + contextual_vector  # 残差连接
        else:
            x = contextual_vector
        # LSTM层 + FC 计算 Q 值
        x = self.fc1(x) + contextual_vector  # 残差连接
        x = F.relu(x) # fc2:[batch_size, hidden_dim
        q = self.fc2(x) #  fc3: [batch_size, 1]
        return q

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file, map_location='cpu'))




class CriticNetwork_2(nn.Module):
    id_num: int = 0
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims,
                 n_agents, n_actions, name, chkpt_dir, dim_info, agent_id, hidden_dim, lstm_hidden_dim, is_recurrent=True, num_layers=8):
        super(CriticNetwork_2, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)
        # 智能体编号
        self.recurrent = is_recurrent
        self.id = CriticNetwork_2.id_num
        CriticNetwork_2.id_num += 1
        # print(CriticNetwork.id_num)
        # 检查 dim_info 的键数量是否足够
        if self.id >= len(dim_info):
            raise ValueError(f"agent_id {self.id} is out of range for dim_info with {len(dim_info)} keys.")

        # 动作维度序号 # [1,2,1] -> [0,1,3,4] # 海象运算符:= 既计算某个值，也将其赋给一个变量
        a_d = [dim_info[id][1] for id in dim_info.keys()]
        sum_ = 0
        self.a_d = [0] + [sum_ := sum_ + i for i in a_d]  # self.a_d 和 self.s_d 用于forward中切片
        # 智能体个数
        self.agent_n = len(dim_info)
        agent_id = list(dim_info.keys())[self.id]

        # 注意力相关
        ''' 这里是每个智能体给一个attention模块'''
        self.attention_modules = Attention_ATT_2(hidden_dim, hidden_dim, hidden_dim, 4)
        # 输入准备
        ## 所有智能体的状态和当前智能体动作 维度
        self.encoder_input_dim = sum([dim_info[agent_id_][0] for agent_id_ in dim_info.keys()]) + dim_info[agent_id][1]
        self.encoder_input_fc = T.nn.Linear(self.encoder_input_dim, hidden_dim)
        ## 其余智能体的动作 维度
        self.decoder_input_dim = sum([dim_info[agent_id_][1] for agent_id_ in dim_info.keys() if agent_id_ != agent_id])
        self.decoder_input_fc = T.nn.Linear(self.decoder_input_dim, hidden_dim)

        # q_value 输出

        if self.recurrent:
            self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True, num_layers=num_layers)
            self.fc1 = nn.Linear(lstm_hidden_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, 1)
        else:
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.7)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    # def forward(self, state, action):
    #     x = F.relu(self.fc1(T.cat([state, action], dim=1)))
    #     x = F.relu(self.fc2(x))
    #     q = self.q(x)
    #
    #     return q

    def forward(self, s, a, hidden_state):
        # s: 拼接后的状态张量，形状 [batch_size, state_dim]
        # a: 拼接后的动作张量，形状 [batch_size, action_dim]
        # hidden_state : 隐藏层状态，形状 [batch_size, lstm_hidden_dim]
        # 当前智能体的动作
        a_i = a[:, self.a_d[self.id]:self.a_d[self.id + 1]]  # self.a_d 用于定位当前智能体的动作切片

        # 编码器输入：全局状态 + 当前智能体动作
        encoder_input = self.encoder_input_fc(T.cat((s, a_i), dim=1))  # [batch_size, state_dim + action_dim]

        # 解码器输入：其他智能体的动作
        decoder_input = self.decoder_input_fc(
            T.cat([a[:, self.a_d[i]:self.a_d[i + 1]] for i in range(self.agent_n) if i != self.id], dim=1)
        )  # [batch_size, action_dim * (agent_count - 1)]

        # 注意力模块
        contextual_vector = self.attention_modules(encoder_input, decoder_input)  # [batch_size, hidden_dim]
        if self.recurrent:
            contextual_vector = contextual_vector.unsqueeze(1)  # 如果是单步输入，添加时间维度
            self.lstm.flatten_parameters()
            x, hidden_state = self.lstm(contextual_vector, hidden_state) # x = [batch_size, lstm_hidden_dim]
        # 加入残差连接
        else:
            x = contextual_vector
        x = self.fc1(x)  # 残差连接
        x = F.relu(x) # fc2:[batch_size, hidden_dim]
        q = self.fc2(x) #  fc3: [batch_size, 1]

        return q  # 输出形状 [batch_size, 1]

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file, map_location='cpu'))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions, name, chkpt_dir,lstm_hidden_dim, is_recurrent = True, num_layers=8):
        super(ActorNetwork, self).__init__()
        self.is_recurrent = is_recurrent
        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.input_dims = input_dims
        self.lstm_hidden_dim = lstm_hidden_dim
        if self.is_recurrent:
            # print("yes")
            self.lstm = nn.LSTM(input_dims, lstm_hidden_dim, batch_first=True, num_layers=num_layers)
            self.fc1 = nn.Linear(lstm_hidden_dim, fc1_dims)
            self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        else:
            self.fc1 = nn.Linear(input_dims, fc1_dims)
            self.fc2 = nn.Linear(fc1_dims, fc2_dims)

        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.7)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, hidden_state):
        # x = F.leaky_relu(self.fc1(state))
        self.lstm.flatten_parameters()  # 防止多 GPU 的问题
        x, hidden = self.lstm(state, hidden_state)  # LSTM 前向传播
        # print(hidden_state)
        # print("state",x.shape)
        x = self.fc1(x)
        x = F.leaky_relu(self.fc2(x))

        pi = nn.Softsign()(self.pi(x))  # [-1,1]
        # print(pi)
        return pi, hidden

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file, map_location='cpu'))



