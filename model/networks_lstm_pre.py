import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims,
                 n_agents, n_actions, name, chkpt_dir, lstm_hidden_dim, num_layers=8):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.lstm = nn.LSTM(input_dims + n_agents * n_actions, lstm_hidden_dim, batch_first=True, num_layers=num_layers)
        self.fc1 = nn.Linear(lstm_hidden_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.7)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action, hidden_state):
        # x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        # x = F.relu(self.fc2(x))
        # q = self.q(x)
        lstm_input = T.cat([state, action], dim=1).unsqueeze(1)
        x, hidden_state = self.lstm(lstm_input, hidden_state)  # x = [batch_size, lstm_hidden_dim]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q(x)  # fc3: [batch_size, 1]
        return q

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file, map_location='cpu'))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions, name, chkpt_dir, lstm_hidden_dim, num_layers=8):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.lstm = nn.LSTM(input_dims, lstm_hidden_dim, batch_first=True, num_layers=num_layers)
        self.fc1 = nn.Linear(lstm_hidden_dim, fc1_dims)
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

