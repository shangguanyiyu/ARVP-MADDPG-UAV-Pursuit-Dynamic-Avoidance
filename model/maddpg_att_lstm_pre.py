import os
import torch as T
from torch import nn
import torch.nn.functional as F
from agent.agent_att_lstm_pre import Agent
# from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from ATT import ATT_critic, ATT_critic_raw
import numpy as np


class MADDPGWithAttentionLSTMPRE:
    """

    actor_dims 和 critic_dims 分别是每个agent的actor网络和评论家网络的输入维度。
    n_agents 是代理的数量。
    n_actions 是每个代理的动作数量。
    scenario 是场景名称，默认为 'simple'。
    alpha 和 beta 分别是演员网络和评论家网络的学习率。
    fc1 和 fc2 是演员网络和评论家网络的隐藏层大小。
    gamma 是折扣因子。
    tau 是目标网络更新的软更新参数。
    chkpt_dir 是检查点保存的目录。
    """
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions,
                 scenario='simple',  alpha=0.01, beta=0.02, fc1=128,
                 fc2=128, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpgwithatt/', act_dim = 2, obs_agt = 26, obs_tar = 23, lstm_hidden_dim = 256, batch_size=256, num_layers=8):
        self.agents = [] # 声明 agent 列表
        self.n_agents = n_agents
        self.n_actions = n_actions
        # self.actor_schedulers = [lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.95) for optimizer in
        #                          self.actor_optimizer]
        # self.critic_schedulers = [lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.95) for optimizer in
        #                           self.critic_optimizer]
        chkpt_dir += scenario
        # self.writer = SummaryWriter(log_dir=os.path.join(chkpt_dir, 'logs'))

        # 初始化dim
        self.dim_info = {}
        for agent_idx in range(self.n_agents - 1):
            self.dim_info[agent_idx] = [obs_agt, act_dim]
        self.dim_info[3] = [obs_tar, act_dim]  # 添加目标的观测和动作维度
        dim_info = self.dim_info
        # self.indices = np.random.randint(0, 256)
        # print(dim_info)
        # self.TD_error = np.random.rand(batch_size)
        # print(self.TD_error.shape)
        # 初始化 hidden_states 和 next_hidden_states
        # self.hidden_states = [(T.zeros((1, lstm_hidden_dim)), T.zeros((1, lstm_hidden_dim))) for _ in
        #                       range(self.n_agents)]
        # self.next_hidden_states = [(T.zeros((1, lstm_hidden_dim)), T.zeros((1, lstm_hidden_dim))) for _ in
        #                            range(self.n_agents)]
        self.hidden_states = None
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,
                            n_actions, n_agents,agent_idx, alpha=alpha, beta=beta,
                            chkpt_dir=chkpt_dir, dim_info=dim_info, agent_id=agent_idx, lstm_hidden_dim = lstm_hidden_dim,num_layers=num_layers))


    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            os.makedirs(os.path.dirname(agent.actor.chkpt_file), exist_ok=True)
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()


    def choose_action(self, raw_obs, time_step, evaluate, hidden_states):# timestep for exploration
        """
        根据当前观测选择动作。

        该方法主要用于在给定的时间步和评估状态下，根据每个代理的观测，为每个代理选择最合适的动作。

        参数:
        - raw_obs: 包含所有代理观测的列表。每个代理的观测可能包括环境状态、其他代理的位置等信息。
        - time_step: 当前的时间步，用于探索策略的调整。随着时间的推移，探索的程度可能会减少，以平衡探索与利用。
        - evaluate: 布尔值，指示是否为评估模式。在评估模式下，代理的行为选择可能更加保守，专注于获得最高奖励。

        返回:
        - actions: 包含所有代理所选动作的列表。每个动作是相应代理根据其观测和当前策略选择的。
        """
        actions = []
        # hiddens = []
        next_hiddens = []
        for agent_idx, agent in enumerate(self.agents):
            action, hidden = agent.choose_action(raw_obs[agent_idx],time_step, hidden_state=hidden_states[agent_idx], evaluate=evaluate)
            actions.append(action)
            # hiddens.append(hidden_states[agent_idx])
            next_hiddens.append(hidden)
        return actions, hidden_states, next_hiddens

    def learn(self, memory, total_steps, current_state, hidden_list,steps):
        if not memory.ready():
            return

        # # 调用 随机采样sample_buffer
        if self.hidden_states is None:
            actor_states, states, actions, rewards, \
                actor_new_states, states_, dones, hidden_states, next_hidden_states = memory.sample_buffer_init()
            self.hidden_states = hidden_states
        else:
            #     # [256,] [256,]
            # 调用 优先级采样sample_buffer
            # 从经验回放中采样
            actor_states, states, actions, rewards, \
                actor_new_states, states_, dones, indices, ISWeights, hidden_states, next_hidden_states = memory.sample_buffer(
                [agent.critic if idx < len(self.agents) - 1 else agent.target_critic
                 for idx, agent in enumerate(self.agents)],  # 前 N-1 个使用 critic，最后一个使用 target_critic
                gamma=self.agents[0].gamma,  # 假设所有智能体共享相同的 gamma
                hidden_states = self.hidden_states

            )
        self.hidden_states = next_hidden_states
        # self.next_hidden_states = next_hidden_states
        device = self.agents[0].actor.device

        # states = T.tensor(states, dtype=T.float).to(device) # [256，101]
        # actions = T.tensor(actions, dtype=T.float).to(device) # [4,256,2]
        # # actions = T.tensor(np.concatenate(actions, axis=1), dtype=T.float).to(device) # [256,8]
        # # print(actions.shape)
        # rewards = T.tensor(rewards, dtype=T.float).to(device) # [256, 4]
        # states_ = T.tensor(states_, dtype=T.float).to(device) # [256, 101]
        # dones = T.tensor(dones).to(device)

        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        # # 隐层状态转换为张量
        # h, c = hidden_states  # 解压当前隐藏状态 (h, c)
        # nh, nc = next_hidden_states  # 解压下一步隐藏状态 (nh, nc)


        # h = T.tensor(h, dtype=T.float).to(device)[None,:,:]
        # c = T.tensor(c, dtype=T.float).to(device)[None,:,:]
        # nh = T.tensor(nh, dtype=T.float).to(device)[None,:,:]
        # nc = T.tensor(nc, dtype=T.float).to(device)[None,:,:]
        # hidden_states = (h, c)
        # next_hidden_states = (nh, nc)

        all_agents_new_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            # 隐层状态转换为张量
            (h, c) = hidden_states[agent_idx]  # 解压当前隐藏状态 (h, c)
            (nh, nc) = next_hidden_states[agent_idx]  # 解压下一步隐藏状态 (nh, nc)

            h = T.tensor(h, dtype=T.float).to(device)
            c = T.tensor(c, dtype=T.float).to(device)
            nh = T.tensor(nh, dtype=T.float).to(device)
            nc = T.tensor(nc, dtype=T.float).to(device)
            saved_hidden_states = (h, c)
            saved_next_hidden_states = (nh, nc)

            new_states = T.tensor(actor_new_states[agent_idx],
                                  dtype=T.float).to(device)[:,None,:] # [1,256,26]
            # print(new_states.shape)

            new_pi, new_hidden_states = agent.target_actor.forward(new_states, saved_next_hidden_states)
            new_pi = new_pi.squeeze()
            # print(new_pi.size())
            all_agents_new_actions.append(new_pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)

        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)
        td_errors = []
        for agent_idx, agent in enumerate(self.agents):
            with T.no_grad():
                # states_ = states_[:,None,:]
                # new_actions = new_actions.unsqueeze(0)
                critic_value_ = agent.target_critic.forward(states_, new_actions, saved_next_hidden_states).flatten()
                target = rewards[:, agent_idx] + (1 - dones[:, 0].int()) * agent.gamma * critic_value_

            critic_value = agent.critic.forward(states, old_actions, saved_hidden_states).flatten()
            td_error = critic_value_ - critic_value
            td_errors.append(td_error.detach().cpu().numpy())  # 转换为 numpy 数组

            # self.indices = indices
            # self.TD_error = max_td_error


            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()
            agent.critic.scheduler.step()

            mu_states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device)[:,None,:]
            oa = old_actions.clone()
            # oa[:, agent_idx * self.n_actions:agent_idx * self.n_actions + self.n_actions] = agent.actor.forward(
            #     mu_states,hidden_states)[0]

            # 获取 actor.forward 的输出，并调整维度
            output = agent.actor.forward(mu_states, saved_hidden_states)[0]  # 假设 forward 返回 (output, hidden_state)
            output = output.squeeze(1)  # 去掉第二维度为 1 的维度，变为 [256, 2]

            # 将调整后的输出赋值给 oa 的切片
            oa[:, agent_idx * self.n_actions:agent_idx * self.n_actions + self.n_actions] = output
            actor_loss = -T.mean(agent.critic.forward(states, oa, saved_hidden_states).flatten())
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()
            agent.actor.scheduler.step()

            # self.writer.add_scalar(f'Agent_{agent_idx}/Actor_Loss', actor_loss.item(), total_steps)
            # self.writer.add_scalar(f'Agent_{agent_idx}/Critic_Loss', critic_loss.item(), total_steps)

            # for name, param in agent.actor.named_parameters():
            #     if param.grad is not None:
            #         self.writer.add_histogram(f'Agent_{agent_idx}/Actor_Gradients/{name}', param.grad, total_steps)
            # for name, param in agent.critic.named_parameters():
            #     if param.grad is not None:
            #         self.writer.add_histogram(f'Agent_{agent_idx}/Critic_Gradients/{name}', param.grad, total_steps)

        for agent in self.agents:
            agent.update_network_parameters()





