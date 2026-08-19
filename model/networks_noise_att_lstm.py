"""
networks_noise_att_lstm.py (trae/arvpNoise 新增)
==================================================
在原 model/networks_att_2.py 基础上:
  - Actor 输入维度从 26 扩展到 42 (26 原 obs + 16 信号)
  - 新增 SignalEncoder (FFT + GRU + 多头自注意力) 对 16 维时域信号分支处理
  - Critic 在原 ATT_critic 基础上,额外把 3 个围捕者的信号分支
    分别送入共享 SignalEncoder 后拼接, 与 ATT 上下文向量融合输出 Q
  - 不改动 reward 机制, 仅是观测/网络结构的扩展
"""
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ATT import Attention_ATT_2
from model.signal_encoder import SignalEncoder


# ---------- 围捕者(hunter)的 Actor ----------
class HunterActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions, name, chkpt_dir,
                 n_signal_samples=16, signal_hidden=64, base_obs_dim=26):
        super(HunterActorNetwork, self).__init__()
        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.n_signal_samples = n_signal_samples
        self.base_obs_dim = base_obs_dim   # 26 维原 obs (无信号)
        self.signal_hidden = signal_hidden

        # 原 obs 编码分支
        self.fc1 = nn.Linear(base_obs_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)

        # 信号分支: FFT + GRU + 自注意力
        self.signal_encoder = SignalEncoder(
            n_samples=n_signal_samples,
            hidden_dim=signal_hidden,
            n_gru_layers=2,
            n_heads=4,
        )

        # 融合: base_obs 特征 + 信号 context
        self.fuse = nn.Linear(fc2_dims + signal_hidden, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.7)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # state: [B, input_dims=42]  末尾 16 维是信号采样
        base_obs = state[:, :self.base_obs_dim]
        signal = state[:, self.base_obs_dim:]
        x = F.leaky_relu(self.fc1(base_obs))
        x = F.leaky_relu(self.fc2(x))
        sig_ctx = self.signal_encoder(signal)             # [B, signal_hidden]
        fused = T.cat([x, sig_ctx], dim=1)
        fused = F.leaky_relu(self.fuse(fused))
        pi = nn.Softsign()(self.pi(fused))                 # [-1, 1]
        return pi

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file, map_location='cpu'))


# ---------- 目标(target, 逃跑者)的 Actor (无信号分支) ----------
class TargetActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions, name, chkpt_dir):
        super(TargetActorNetwork, self).__init__()
        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.7)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        pi = nn.Softsign()(self.pi(x))
        return pi

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file, map_location='cpu'))


# ---------- Critic 网络 (含信号分支) ----------
class CriticNetwork_1(nn.Module):
    id_num: int = 0

    def __init__(self, beta, input_dims, fc1_dims, fc2_dims,
                 n_agents, n_actions, name, chkpt_dir, dim_info, agent_id,
                 hidden_dim, n_signal_samples=16, signal_hidden=64,
                 obs_agt=42, obs_tar=23, n_hunters=3):
        super(CriticNetwork_1, self).__init__()
        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.id = CriticNetwork_1.id_num
        CriticNetwork_1.id_num += 1
        if self.id >= len(dim_info):
            raise ValueError(f"agent_id {self.id} out of range for dim_info with {len(dim_info)} keys.")

        a_d = [dim_info[id][1] for id in dim_info.keys()]
        sum_ = 0
        self.a_d = [0] + [sum_ := sum_ + i for i in a_d]
        self.agent_n = len(dim_info)
        agent_id = list(dim_info.keys())[self.id]

        # ATT 注意力 (原结构)
        self.attention_modules = Attention_ATT_2(hidden_dim, hidden_dim, hidden_dim, 4)
        self.encoder_input_dim = sum([dim_info[agent_id_][0] for agent_id_ in dim_info.keys()]) + dim_info[agent_id][1]
        self.encoder_input_fc = T.nn.Linear(self.encoder_input_dim, hidden_dim)
        self.decoder_input_dim = sum([dim_info[agent_id_][1] for agent_id_ in dim_info.keys() if agent_id_ != agent_id])
        self.decoder_input_fc = T.nn.Linear(self.decoder_input_dim, hidden_dim)

        # 信号分支: 为 3 个围捕者各设一个共享 SignalEncoder
        self.n_signal_samples = n_signal_samples
        self.n_hunters = n_hunters
        self.obs_agt = obs_agt          # 42
        self.obs_tar = obs_tar          # 23
        self.signal_encoders = nn.ModuleList([
            SignalEncoder(n_samples=n_signal_samples,
                          hidden_dim=signal_hidden,
                          n_gru_layers=2, n_heads=4)
            for _ in range(n_hunters)
        ])
        # ATT context(hidden_dim) + 3 * signal_hidden
        self.fuse = nn.Linear(hidden_dim + signal_hidden * n_hunters, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.7)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, s, a):
        # s: [B, state_dim]  全局状态(包含每个 agent 的 obs 拼接)
        # a: [B, action_dim] 全局动作
        a_i = a[:, self.a_d[self.id]:self.a_d[self.id + 1]]
        encoder_input = self.encoder_input_fc(T.cat((s, a_i), dim=1))
        decoder_input = self.decoder_input_fc(
            T.cat([a[:, self.a_d[i]:self.a_d[i + 1]] for i in range(self.agent_n) if i != self.id], dim=1)
        )
        contextual_vector = self.attention_modules(encoder_input, decoder_input)  # [B, hidden_dim]

        # 从全局状态 s 中切片出每个围捕者的信号部分 (末 16 维)
        sig_feats = []
        for k in range(self.n_hunters):
            start = k * self.obs_agt + (self.obs_agt - self.n_signal_samples)
            end   = k * self.obs_agt + self.obs_agt
            sig_k = s[:, start:end]
            sig_feats.append(self.signal_encoders[k](sig_k))    # [B, signal_hidden]
        sig_cat = T.cat(sig_feats, dim=1)                       # [B, signal_hidden*3]

        fused = self.fuse(T.cat([contextual_vector, sig_cat], dim=1))
        x = F.relu(self.fc1(fused))
        q = self.fc2(x)
        return q

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file, map_location='cpu'))


class CriticNetwork_2(nn.Module):
    id_num: int = 0

    def __init__(self, beta, input_dims, fc1_dims, fc2_dims,
                 n_agents, n_actions, name, chkpt_dir, dim_info, agent_id,
                 hidden_dim, n_signal_samples=16, signal_hidden=64,
                 obs_agt=42, obs_tar=23, n_hunters=3):
        super(CriticNetwork_2, self).__init__()
        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.id = CriticNetwork_2.id_num
        CriticNetwork_2.id_num += 1
        if self.id >= len(dim_info):
            raise ValueError(f"agent_id {self.id} out of range for dim_info with {len(dim_info)} keys.")

        a_d = [dim_info[id][1] for id in dim_info.keys()]
        sum_ = 0
        self.a_d = [0] + [sum_ := sum_ + i for i in a_d]
        self.agent_n = len(dim_info)
        agent_id = list(dim_info.keys())[self.id]

        self.attention_modules = Attention_ATT_2(hidden_dim, hidden_dim, hidden_dim, 4)
        self.encoder_input_dim = sum([dim_info[agent_id_][0] for agent_id_ in dim_info.keys()]) + dim_info[agent_id][1]
        self.encoder_input_fc = T.nn.Linear(self.encoder_input_dim, hidden_dim)
        self.decoder_input_dim = sum([dim_info[agent_id_][1] for agent_id_ in dim_info.keys() if agent_id_ != agent_id])
        self.decoder_input_fc = T.nn.Linear(self.decoder_input_dim, hidden_dim)

        self.n_signal_samples = n_signal_samples
        self.n_hunters = n_hunters
        self.obs_agt = obs_agt
        self.obs_tar = obs_tar
        self.signal_encoders = nn.ModuleList([
            SignalEncoder(n_samples=n_signal_samples,
                          hidden_dim=signal_hidden,
                          n_gru_layers=2, n_heads=4)
            for _ in range(n_hunters)
        ])
        self.fuse = nn.Linear(hidden_dim + signal_hidden * n_hunters, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.7)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, s, a):
        a_i = a[:, self.a_d[self.id]:self.a_d[self.id + 1]]
        encoder_input = self.encoder_input_fc(T.cat((s, a_i), dim=1))
        decoder_input = self.decoder_input_fc(
            T.cat([a[:, self.a_d[i]:self.a_d[i + 1]] for i in range(self.agent_n) if i != self.id], dim=1)
        )
        contextual_vector = self.attention_modules(encoder_input, decoder_input)

        sig_feats = []
        for k in range(self.n_hunters):
            start = k * self.obs_agt + (self.obs_agt - self.n_signal_samples)
            end   = k * self.obs_agt + self.obs_agt
            sig_k = s[:, start:end]
            sig_feats.append(self.signal_encoders[k](sig_k))
        sig_cat = T.cat(sig_feats, dim=1)

        fused = self.fuse(T.cat([contextual_vector, sig_cat], dim=1))
        x = F.relu(self.fc1(fused))
        q = self.fc2(x)
        return q

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file, map_location='cpu'))
