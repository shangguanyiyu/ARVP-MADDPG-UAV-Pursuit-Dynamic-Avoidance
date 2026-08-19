"""
maddpg_noise_att_lstm.py (trae/arvpNoise 新增)
==================================================
基于 model/maddpg_att.py 改造:
  - 使用新 Agent (含 SignalEncoder 分支处理 16 维时域声呐采样)
  - obs_agt 默认从 26 扩到 42, obs_tar 仍为 23
  - 复用 PERMultiAgentReplayBuffer 和原 ATT critic 流程
  - 不修改 env 的 reward 机制
"""
import os
import torch as T
from torch import nn
import torch.nn.functional as F
from agent.agent_noise_att_lstm import Agent


class MADDPGNoiseWithAttention:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions,
                 scenario='simple', alpha=0.01, beta=0.02, gamma=0.99,
                 chkpt_dir='tmp_noise/maddpgwithatt/',
                 obs_agt=42, obs_tar=23,
                 n_signal_samples=16, signal_hidden=64, n_hunters=3):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.n_signal_samples = n_signal_samples
        chkpt_dir += scenario

        # dim_info: 围捕者 obs_agt 维, 目标 obs_tar 维, 每个动作 n_actions 维
        self.dim_info = {}
        for agent_idx in range(self.n_agents - 1):
            self.dim_info[agent_idx] = [obs_agt, n_actions]
        self.dim_info[self.n_agents - 1] = [obs_tar, n_actions]
        dim_info = self.dim_info

        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(
                actor_dims[agent_idx], critic_dims,
                n_actions, n_agents, agent_idx,
                alpha=alpha, beta=beta, gamma=gamma,
                chkpt_dir=chkpt_dir, dim_info=dim_info, agent_id=agent_idx,
                n_signal_samples=n_signal_samples, signal_hidden=signal_hidden,
                obs_agt=obs_agt, obs_tar=obs_tar, n_hunters=n_hunters,
            ))

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            os.makedirs(os.path.dirname(agent.actor.chkpt_file), exist_ok=True)
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs, time_step, evaluate):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx], time_step, evaluate)
            actions.append(action)
        return actions

    def learn(self, memory, total_steps, current_state):
        if not memory.ready():
            return
        # PER 采样 (与原 maddpg_att.py 流程一致)
        actor_states, states, actions, rewards, \
            actor_new_states, states_, dones, indices, ISWeights = memory.sample_buffer(
            [agent.critic if idx < len(self.agents) - 1 else agent.target_critic
             for idx, agent in enumerate(self.agents)],
            gamma=self.agents[0].gamma,
        )
        device = self.agents[0].actor.device

        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        old_agents_actions = []
        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx], dtype=T.float).to(device)
            new_pi = agent.target_actor.forward(new_states)
            all_agents_new_actions.append(new_pi)
            old_agents_actions.append(actions[agent_idx])
        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)

        for agent_idx, agent in enumerate(self.agents):
            with T.no_grad():
                critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
                target = rewards[:, agent_idx] + (1 - dones[:, 0].int()) * agent.gamma * critic_value_
            critic_value = agent.critic.forward(states, old_actions).flatten()
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            mu_states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device)
            oa = old_actions.clone()
            oa[:, agent_idx * self.n_actions:agent_idx * self.n_actions + self.n_actions] = agent.actor.forward(mu_states)
            actor_loss = -T.mean(agent.critic.forward(states, oa).flatten())
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

        for agent in self.agents:
            agent.update_network_parameters()
