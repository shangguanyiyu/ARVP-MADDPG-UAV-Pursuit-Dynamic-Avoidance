from maddpg_lstm_pre import MADDPGWithAttentionLSTMPRE
from sim_env_rvo import UAVEnv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings
import os
from UAVTask import UAVTaskEvaluator
import torch

warnings.filterwarnings('ignore')


def moving_average(data, window_size=5):
    """计算滑动窗口均值"""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def plot_velocities(velocities_magnitude, velocities_x, velocities_y):
    """绘制速度信息"""
    time_steps = range(len(velocities_magnitude[0]))
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    for i in range(len(velocities_magnitude)):
        axs[0].plot(time_steps, velocities_magnitude[i], label=f'UAV {i}')
    axs[0].set_title('Speed Magnitude vs Time')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Speed Magnitude')
    axs[0].legend()

    for i in range(len(velocities_x)):
        axs[1].plot(time_steps, velocities_x[i], label=f'UAV {i}')
    axs[1].set_title('Velocity X Component vs Time')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Velocity X Component')
    axs[1].legend()

    for i in range(len(velocities_y)):
        axs[2].plot(time_steps, velocities_y[i], label=f'UAV {i}')
    axs[2].set_title('Velocity Y Component vs Time')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Velocity Y Component')
    axs[2].legend()

    plt.tight_layout()
    plt.show()


def init_hidden_memory( num_layers,batch_size, hidden_size, device):
    h_0 = torch.zeros((num_layers, batch_size, hidden_size), dtype=torch.float).to(device)
    c_0 = torch.zeros((num_layers, batch_size, hidden_size), dtype=torch.float).to(device)
    return h_0, c_0


if __name__ == '__main__':
    # 初始化环境和模型
    env = UAVEnv()
    n_agents = env.num_agents
    n_actions = 2
    actor_dims = []
    lstm_hidden_dim = 256

    velocities_magnitude = [[] for _ in range(env.num_agents)]  # 记录速度大小
    velocities_x = [[] for _ in range(env.num_agents)]  # 记录速度 x 分量
    velocities_y = [[] for _ in range(env.num_agents)]  # 记录速度 y 分量
    trajectories = [[] for _ in range(env.num_agents)]  # 每个无人机的轨迹
    collisions_record = [[] for _ in range(env.num_agents)]  # 每个无人机的碰撞记录

    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 初始化模型
    maddpg_agents = MADDPGWithAttentionLSTMPRE(actor_dims, critic_dims, n_agents, n_actions,
                                            fc1=128, fc2=128, lstm_hidden_dim=lstm_hidden_dim,
                                            alpha=0.0001, beta=0.001, scenario='UAV_Round_up',
                                            chkpt_dir='tmp_test/maddpgwithatt/')
    maddpg_agents.load_checkpoint()
    print('---- Evaluating ----')


    obs = env.reset()
    hidden_state_actor  = [init_hidden_memory(2, 1, lstm_hidden_dim, device=device) for i in range(env.num_agents)]

    def update(frame):
        """更新动画帧"""
        global obs, hidden_state_actor, velocities_magnitude, velocities_x, velocities_y
        global trajectories, collisions_record, total_steps

        # 选择动作并与环境交互
        actions, hidden_state_actor, next_hidden_state_actor_list = maddpg_agents.choose_action(obs, total_steps, evaluate=True, hidden_states=hidden_state_actor)
        obs_, rewards, dones, collision_info, mul_pos = env.step(actions)

        for i in range(env.num_agents):
            # 记录轨迹
            trajectories[i].append(env.multi_current_pos[i])

            # 记录碰撞信息
            collisions_record[i].append(collision_info)

            # 记录速度信息
            vel = env.multi_current_vel[i]
            v_x, v_y = vel
            speed = np.linalg.norm(vel)
            velocities_magnitude[i].append(speed)
            velocities_x[i].append(v_x)
            velocities_y[i].append(v_y)

        # 渲染动画帧
        env.render_anime(frame)
        obs = obs_
        hidden_state_actor = next_hidden_state_actor_list

        # 终止条件
        if any(dones) or frame > 1000:
            ani.event_source.stop()
            print("Round-up finished in", frame, "steps.")

            # 评测结果
            evaluator = UAVTaskEvaluator([0, 0, 0], max_time_steps=1000, dones=dones)
            for i in range(env.num_agents):
                result = evaluator.evaluate(trajectories[i], collisions_record[i], mul_pos)
                print(f"评测结果 (UAV {i}):")
                for key, value in result.items():
                    print(f"  {key}: {value:.2f}")

            # 绘制速度信息
            smoothed_velocities_magnitude = [moving_average(v, window_size=5) for v in velocities_magnitude]
            smoothed_velocities_x = [moving_average(v, window_size=5) for v in velocities_x]
            smoothed_velocities_y = [moving_average(v, window_size=5) for v in velocities_y]
            plot_velocities(smoothed_velocities_magnitude, smoothed_velocities_x, smoothed_velocities_y)

        total_steps += 1
        return []

    total_steps = 0

    # 创建动画
    fig = plt.figure()
    ani = animation.FuncAnimation(fig, update, frames=10000, interval=20)
    plt.show()
