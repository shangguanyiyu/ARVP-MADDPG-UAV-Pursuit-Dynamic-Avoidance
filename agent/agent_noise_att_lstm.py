"""
agent_noise_att_lstm.py (trae/arvpNoise 新增)
================================================
- 围捕者(agent_0/1/2) 使用 HunterActorNetwork (含 SignalEncoder 分支)
- 目标(agent_3)     使用 TargetActorNetwork (无信号分支)
- Critic 使用新 CriticNetwork_1 / CriticNetwork_2 (含信号分支)
- 复用原 ATT/PER 训练机制
"""
import torch as T
import numpy as np
from model.networks_noise_att_lstm import (
    HunterActorNetwork,
    TargetActorNetwork,
    CriticNetwork_1,
    CriticNetwork_2,
)


class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx,
                 chkpt_dir, dim_info, agent_id,
                 alpha=0.0001, beta=0.0002, fc1=128, fc2=128,
                 gamma=0.99, tau=0.01,
                 n_signal_samples=16, signal_hidden=64,
                 obs_agt=42, obs_tar=23, n_hunters=3):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        self.is_hunter = (agent_idx < n_hunters)
        # Actor: 围捕者用 Hunter (含信号分支), 目标用 Target
        if self.is_hunter:
            self.actor = HunterActorNetwork(
                alpha, actor_dims, fc1, fc2, n_actions,
                chkpt_dir=chkpt_dir, name=self.agent_name + '_actor',
                n_signal_samples=n_signal_samples, signal_hidden=signal_hidden,
                base_obs_dim=actor_dims - n_signal_samples,
            )
            self.target_actor = HunterActorNetwork(
                alpha, actor_dims, fc1, fc2, n_actions,
                chkpt_dir=chkpt_dir, name=self.agent_name + '_target_actor',
                n_signal_samples=n_signal_samples, signal_hidden=signal_hidden,
                base_obs_dim=actor_dims - n_signal_samples,
            )
        else:
            self.actor = TargetActorNetwork(
                alpha, actor_dims, fc1, fc2, n_actions,
                chkpt_dir=chkpt_dir, name=self.agent_name + '_actor',
            )
            self.target_actor = TargetActorNetwork(
                alpha, actor_dims, fc1, fc2, n_actions,
                chkpt_dir=chkpt_dir, name=self.agent_name + '_target_actor',
            )

        self.critic = CriticNetwork_1(
            beta, critic_dims, fc1, fc2, n_agents, n_actions,
            chkpt_dir=chkpt_dir, name=self.agent_name + '_critic',
            dim_info=dim_info, agent_id=agent_id, hidden_dim=128,
            n_signal_samples=n_signal_samples, signal_hidden=signal_hidden,
            obs_agt=obs_agt, obs_tar=obs_tar, n_hunters=n_hunters,
        )
        self.target_critic = CriticNetwork_2(
            beta, critic_dims, fc1, fc2, n_agents, n_actions,
            chkpt_dir=chkpt_dir, name=self.agent_name + '_target_critic',
            dim_info=dim_info, agent_id=agent_id, hidden_dim=128,
            n_signal_samples=n_signal_samples, signal_hidden=signal_hidden,
            obs_agt=obs_agt, obs_tar=obs_tar, n_hunters=n_hunters,
        )

        # reset 类级静态 id 计数 (CriticNetwork_X.id_num) 必须在创建下一个 agent 前重置
        # 注意: 由于 CriticNetwork_1/2 共享类变量 id_num, 但它们的 id 流程独立
        # 上面创建 4 个 agent 时, CriticNetwork_1.id_num 会 0,1,2,3
        # 而 CriticNetwork_2.id_num 也 0,1,2,3 — 与现有代码模式一致

        self.update_network_parameters(tau=1)

    def choose_action(self, observation, time_step, evaluate=False):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        actions = self.actor.forward(state)
        max_noise = 0.75
        min_noise = 0.03
        decay_rate = 0.9999995
        noise_scale = max(min_noise, max_noise * (decay_rate ** time_step))
        noise = 2 * T.rand(self.n_actions).to(self.actor.device) - 1
        if not evaluate:
            noise = noise_scale * noise
        else:
            noise = 0 * noise
        action = actions + noise
        action_np = action.detach().cpu().numpy()[0]
        magnitude = np.linalg.norm(action_np)
        if magnitude > 0.04:
            action_np = action_np / magnitude * 0.04
        return action_np

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        # actor
        target_actor_state_dict = dict(self.target_actor.named_parameters())
        actor_state_dict = dict(self.actor.named_parameters())
        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                (1 - tau) * target_actor_state_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)
        # critic
        target_critic_state_dict = dict(self.target_critic.named_parameters())
        critic_state_dict = dict(self.critic.named_parameters())
        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                (1 - tau) * target_critic_state_dict[name].clone()
        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
