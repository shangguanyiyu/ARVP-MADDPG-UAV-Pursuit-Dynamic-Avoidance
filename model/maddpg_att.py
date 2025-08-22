import os
import torch as T
from torch import nn
import torch.nn.functional as F
from agent.agent_att import Agent
# from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from ATT import ATT_critic, ATT_critic_raw


class MADDPGWithAttention:
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
                 scenario='simple',  alpha=0.01, beta=0.02, gamma = 0.99,chkpt_dir='tmp_first/maddpgwithatt/', obs_agt = 26, obs_tar = 23):
        self.agents = [] # 声明 agent 列表
        self.n_agents = n_agents
        self.n_actions = n_actions

        chkpt_dir += scenario
        # self.writer = SummaryWriter(log_dir=os.path.join(chkpt_dir, 'logs'))

        # 初始化dim
        self.dim_info = {}
        for agent_idx in range(self.n_agents - 1):
            self.dim_info[agent_idx] = [obs_agt, n_actions]
        self.dim_info[3] = [obs_tar, n_actions]  # 添加目标的观测和动作维度
        dim_info = self.dim_info
        # print(dim_info)
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,  
                            n_actions, n_agents,agent_idx, alpha=alpha, beta=beta,gamma = gamma,
                            chkpt_dir=chkpt_dir, dim_info=dim_info, agent_id=agent_idx))



        # 初始化每个代理的演员网络和评论家网络
        # self.actor_schedulers = [
        #     lr_scheduler.CosineAnnealingLR(agent.actor.optimizer, T_max=100, eta_min=1e-8)
        #     for agent in self.agents
        # ]
        # self.critic_schedulers = [
        #     lr_scheduler.CosineAnnealingLR(agent.critic.optimizer, T_max=100, eta_min=1e-8)
        #     for agent in self.agents
        # ]
        # 初始化余弦退火


    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            os.makedirs(os.path.dirname(agent.actor.chkpt_file), exist_ok=True)
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()


    def choose_action(self, raw_obs, time_step, evaluate):# timestep for exploration
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
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx],time_step, evaluate)
            actions.append(action)
        return actions

    # def learn(self, memory, total_steps, current_state):
    #     if not memory.ready():
    #         return
    #
    #     # 修改为带有优先级和相关性的采样：
    #     # actor_states, states, actions, rewards, \
    #     #     actor_new_states, states_, dones = memory.sample_buffer_with_correlation(current_state)
    #
    #     # actor_states, states, actions, rewards, \
    #     #     actor_new_states, states_, dones = memory.sample_buffer()
    #
    #     # 调用 sample_buffer，获取 TD 误差
    #     actor_states, states, actions, rewards, \
    #         actor_new_states, states_, dones, indices, ISWeights = memory.sample_buffer(
    #         [agent.critic if idx < len(self.agents) - 1 else agent.target_critic
    #          for idx, agent in enumerate(self.agents)],  # 前 N-1 个使用 critic，最后一个使用 target_critic
    #         gamma=self.agents[0].gamma  # 假设所有智能体共享相同的 gamma
    #     )
    #
    #     device = self.agents[0].actor.device
    #
    #     states = T.tensor(states, dtype=T.float).to(device)
    #     actions = T.tensor(actions, dtype=T.float).to(device)
    #     rewards = T.tensor(rewards, dtype=T.float).to(device)
    #     states_ = T.tensor(states_, dtype=T.float).to(device)
    #     dones = T.tensor(dones).to(device)
    #
    #     all_agents_new_actions = []
    #     old_agents_actions = []
    #
    #     for agent_idx, agent in enumerate(self.agents):
    #         new_states = T.tensor(actor_new_states[agent_idx],
    #                               dtype=T.float).to(device)
    #
    #         new_pi = agent.target_actor.forward(new_states)
    #
    #         all_agents_new_actions.append(new_pi)
    #         old_agents_actions.append(actions[agent_idx])
    #
    #     new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
    #     old_actions = T.cat([acts for acts in old_agents_actions], dim=1)
    #
    #     for agent_idx, agent in enumerate(self.agents):
    #         with T.no_grad():
    #             critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
    #             target = rewards[:, agent_idx] + (1 - dones[:, 0].int()) * agent.gamma * critic_value_
    #
    #         critic_value = agent.critic.forward(states, old_actions).flatten()
    #
    #         critic_loss = F.mse_loss(target, critic_value)
    #         agent.critic.optimizer.zero_grad()
    #         critic_loss.backward(retain_graph=True)
    #         agent.critic.optimizer.step()
    #         agent.critic.scheduler.step()
    #
    #         mu_states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device)
    #         oa = old_actions.clone()
    #         oa[:, agent_idx * self.n_actions:agent_idx * self.n_actions + self.n_actions] = agent.actor.forward(
    #             mu_states)
    #         actor_loss = -T.mean(agent.critic.forward(states, oa).flatten())
    #         agent.actor.optimizer.zero_grad()
    #         actor_loss.backward(retain_graph=True)
    #         agent.actor.optimizer.step()
    #         agent.actor.scheduler.step()
    #
    #         # self.writer.add_scalar(f'Agent_{agent_idx}/Actor_Loss', actor_loss.item(), total_steps)
    #         # self.writer.add_scalar(f'Agent_{agent_idx}/Critic_Loss', critic_loss.item(), total_steps)
    #
    #         # for name, param in agent.actor.named_parameters():
    #         #     if param.grad is not None:
    #         #         self.writer.add_histogram(f'Agent_{agent_idx}/Actor_Gradients/{name}', param.grad, total_steps)
    #         # for name, param in agent.critic.named_parameters():
    #         #     if param.grad is not None:
    #         #         self.writer.add_histogram(f'Agent_{agent_idx}/Critic_Gradients/{name}', param.grad, total_steps)
    #
    #     for agent in self.agents:
    #         agent.update_network_parameters()

    def learn(self, memory, total_steps, current_state):
        if not memory.ready():
            return

        # 从经验回放中采样
        actor_states, states, actions, rewards, \
            actor_new_states, states_, dones, indices, ISWeights = memory.sample_buffer(
            [agent.critic if idx < len(self.agents) - 1 else agent.target_critic
             for idx, agent in enumerate(self.agents)],  # 前 N-1 个使用 critic，最后一个使用 target_critic
            gamma=self.agents[0].gamma  # 假设所有智能体共享相同的 gamma
        )
        # actor_states, states, actions, rewards, \
        #     actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[0].actor.device

        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx],
                                  dtype=T.float).to(device)
            # print("new",new_states.shape)
            new_pi = agent.target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)

        for agent_idx, agent in enumerate(self.agents):
            # 更新评论家网络
            with T.no_grad():
                critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
                target = rewards[:, agent_idx] + (1 - dones[:, 0].int()) * agent.gamma * critic_value_

            critic_value = agent.critic.forward(states, old_actions).flatten()

            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            # 调用评论家调度器更新学习率
            # self.critic_schedulers[agent_idx].step()

            # 更新演员网络
            mu_states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device)
            oa = old_actions.clone()
            oa[:, agent_idx * self.n_actions:agent_idx * self.n_actions + self.n_actions] = agent.actor.forward(
                mu_states)
            actor_loss = -T.mean(agent.critic.forward(states, oa).flatten())
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            # 调用演员调度器更新学习率
            # self.actor_schedulers[agent_idx].step()

        # 更新目标网络参数
        for agent in self.agents:
            agent.update_network_parameters()





