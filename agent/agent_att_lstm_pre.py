import torch as T
from model.networks_att_lstm_pre import ActorNetwork, CriticNetwork_1, CriticNetwork_2
import numpy as np

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir, dim_info, agent_id,
                    alpha=0.0001, beta=0.0002, fc1=128,
                    fc2=128, gamma=0.99, tau=0.01, lstm_hidden_dim=256, num_layers=8):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                  chkpt_dir=chkpt_dir,  name=self.agent_name+'_actor', lstm_hidden_dim=lstm_hidden_dim, num_layers=num_layers)
        self.critic = CriticNetwork_1(beta, critic_dims,
                            fc1, fc2, n_agents, n_actions,
                            chkpt_dir=chkpt_dir, name=self.agent_name+'_critic', dim_info=dim_info, agent_id=agent_id, hidden_dim=128,lstm_hidden_dim=lstm_hidden_dim, num_layers=num_layers)
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                        chkpt_dir=chkpt_dir, 
                                        name=self.agent_name+'_target_actor',lstm_hidden_dim=lstm_hidden_dim, num_layers=num_layers)
        self.target_critic = CriticNetwork_2(beta, critic_dims,
                                            fc1, fc2, n_agents, n_actions,
                                            chkpt_dir=chkpt_dir,
                                            name=self.agent_name+'_target_critic', dim_info=dim_info, agent_id=agent_id, hidden_dim=128,lstm_hidden_dim=lstm_hidden_dim, num_layers=num_layers)
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, time_step, hidden_state , evaluate=False):
        """
        根据当前观测值，结合策略网络+噪声输出一个动作
        :param observation: 当前观测值
        :param time_step: 当前时间步，用于计算噪声尺度
        :param evaluate:
        :return:
        """
        # [1, 1, 26]
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)[:,None,:]
        # print(state.shape)
        # print(state.shape)

        # print("Hidden state before actor.forward:", hidden_state[0].shape, hidden_state[1].shape)
        actions, hidden = self.actor.forward(state, hidden_state)
        # print("Hidden state after actor.forward:", hidden[0].shape, hidden[1].shape)
        # print(actions.size())
        actions = actions.squeeze(1)
        # exploration
        max_noise = 0.75
        min_noise = 0.01
        decay_rate = 0.999995

        noise_scale = max(min_noise, max_noise * (decay_rate ** time_step))
        noise = 2 * T.rand(self.n_actions).to(self.actor.device) - 1 # [-1,1)
        if not evaluate:
            noise = noise_scale * noise
        else:
            noise = 0 * noise

        action = actions + noise
        action_np = action.detach().cpu().numpy()[0]
        magnitude = np.linalg.norm(action_np)
        if magnitude > 0.04:
            action_np = action_np / magnitude * 0.04
        return action_np, hidden

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

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

