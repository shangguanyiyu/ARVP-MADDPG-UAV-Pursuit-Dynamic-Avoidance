from model.maddpg_att import MADDPGWithAttention
from model.maddpg import MADDPG
from env.sim_env_rvo import UAVEnv
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings
import os
import time
from UAVTask import UAVTaskEvaluator

warnings.filterwarnings('ignore')


def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_velocity_magnitude(time_steps, velocities_magnitude):
    plt.figure(figsize=(15, 4))
    for i in range(len(velocities_magnitude)):
        if i!=3:
            plt.plot(time_steps, velocities_magnitude[i], label=f'UAV {i}')
        else:
            plt.plot(time_steps, velocities_magnitude[i], label='Target')
    plt.xlabel("Time Steps")
    plt.ylabel("Magnitude")
    plt.title("UAV Velocity Magnitude")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_velocity_x(time_steps, velocities_x):
    plt.figure(figsize=(15, 4))
    for i in range(len(velocities_x)):
        if i!=3:
            plt.plot(time_steps, velocities_x[i], label=f'UAV {i}')
        else:
            plt.plot(time_steps, velocities_x[i], label='Target')
    plt.xlabel("Time Steps")
    plt.ylabel("$vel_x$")
    plt.title("UAV $Vel_x$")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_velocity_y(time_steps, velocities_y):
    plt.figure(figsize=(15, 4))
    for i in range(len(velocities_y)):
        if i!=3:
            plt.plot(time_steps, velocities_y[i], label=f'UAV {i}')
        else:
            plt.plot(time_steps, velocities_y[i], label='Target')
    plt.xlabel("Time Steps")
    plt.ylabel("$vel_y$")
    plt.title("UAV $Vel_y$")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_velocities(velocities_magnitude, velocities_x, velocities_y, velocities_z):
    time_steps = range(len(velocities_magnitude[0]))
    fig, axs = plt.subplots(4, 1, figsize=(10, 12))

    for i in range(len(velocities_magnitude)):
        if i != 3:
            axs[0].plot(time_steps, velocities_magnitude[i], label=f'UAV {i}')
        else:
            axs[0].plot(time_steps, velocities_magnitude[i], label=f'Target')
    axs[0].set_title('Speed Magnitude vs Time')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Speed Magnitude')
    axs[0].legend()

    for i in range(len(velocities_x)):
        if i != 3:
            axs[1].plot(time_steps, velocities_x[i], label=f'UAV {i}')
        else:
            axs[1].plot(time_steps, velocities_x[i], label=f'Target')
    axs[1].set_title('Velocity X Component vs Time')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Velocity X Component')
    axs[1].legend()

    for i in range(len(velocities_y)):
        if i != 3:
            axs[2].plot(time_steps, velocities_y[i], label=f'UAV {i}')
        else:
            axs[2].plot(time_steps, velocities_y[i], label=f'Target')
    axs[2].set_title('Velocity Y Component vs Time')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Velocity Y Component')
    axs[2].legend()

    for i in range(len(velocities_z)):
        if i != 3:
            axs[3].plot(time_steps, velocities_z[i], label=f'UAV {i}')
        else:
            axs[3].plot(time_steps, velocities_z[i], label=f'Target')
    axs[3].set_title('Velocity Z Component vs Time')
    axs[3].set_xlabel('Time Step')
    axs[3].set_ylabel('Velocity Z Component')
    axs[3].legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    env = UAVEnv()
    n_agents = env.num_agents
    n_actions = 3  # 3D actions (ax, ay, az)
    actor_dims = []
    velocities_magnitude = [[] for _ in range(env.num_agents)]  # record magnitude of vel
    velocities_x = [[] for _ in range(env.num_agents)]  # record vel_x
    velocities_y = [[] for _ in range(env.num_agents)]  # record vel_y
    velocities_z = [[] for _ in range(env.num_agents)]  # record vel_z
    trajectories = [[] for _ in range(env.num_agents)]  # 每个无人机的轨迹
    collisions_record = [[] for _ in range(env.num_agents)]  # 每个无人机的碰撞记录
    energy_consumption = [0 for _ in range(env.num_agents)]  # 每个无人机的能量消耗

    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)
    # 3D observation dims: hunter=47, target=41 (derived from env.observation_space)
    obs_agt = actor_dims[0]
    obs_tar = actor_dims[-1]
    maddpg_agents = MADDPGWithAttention(actor_dims, critic_dims, n_agents, n_actions,
                           obs_agt=obs_agt, obs_tar=obs_tar,
                           alpha=0.00001, beta=0.00001, scenario='UAV_Round_up',
                           chkpt_dir='tmp_avoid_dynamic/maddpgwithatt/')

    try:
        maddpg_agents.load_checkpoint()
    except Exception as e:
        print(f'[warn] load_checkpoint skipped (no/incompatible checkpoint): {e}')
    print('---- Evaluating ----')

    obs = env.reset()


    def update(frame):
        global obs, velocities_magnitude, velocities_x, velocities_y, velocities_z
        global trajectories, collisions_record, energy_consumption, total_steps

        actions = maddpg_agents.choose_action(obs, total_steps, evaluate=True)
        obs_, rewards, dones, collision_info, mul_pos = env.step(actions)
        # print(collision_info)

        for i in range(env.num_agents):
            # 记录轨迹（假设env.multi_current_pos[i]为无人机当前位置[x,y,z]）
            trajectories[i].append(env.multi_current_pos[i])

            # 记录碰撞信息（假设env.collision_status[i]为布尔值）
            collisions_record[i].append(collision_info)
            # print(collisions_record[i])


            # 记录能量消耗（假设env.energy_cost[i]为当前步能耗）
            # energy_consumption[i] += env.energy_cost[i]

            # 记录速度信息 (已有)
            vel = env.multi_current_vel[i]
            v_x, v_y, v_z = vel[:3]
            speed = np.linalg.norm(vel)
            velocities_magnitude[i].append(speed)
            velocities_x[i].append(v_x)
            velocities_y[i].append(v_y)
            velocities_z[i].append(v_z)

        env.render_anime(frame)
        obs = obs_

        if any(dones) or frame > 1000:
            ani.event_source.stop()
            print("Round-up finished in", frame, "steps.")

            # 进行评测
            evaluator = UAVTaskEvaluator([0, 0, 0], max_time_steps=1000,dones = dones)  # 根据实际目标位置修改

            for i in range(env.num_agents):
                result = evaluator.evaluate(trajectories[i], collisions_record[i], mul_pos)
                print(f"评测结果 (UAV {i}):")
                for key, value in result.items():
                    print(f"  {key}: {value:.2f}")

            # 平滑数据绘图（可选）
            smoothed_velocities_magnitude = [moving_average(v, window_size=5) for v in velocities_magnitude]
            smoothed_velocities_x = [moving_average(v, window_size=5) for v in velocities_x]
            smoothed_velocities_y = [moving_average(v, window_size=5) for v in velocities_y]
            smoothed_velocities_z = [moving_average(v, window_size=5) for v in velocities_z]
            time_steps = range(len(smoothed_velocities_magnitude[0]))
            plot_velocities(smoothed_velocities_magnitude, smoothed_velocities_x, smoothed_velocities_y, smoothed_velocities_z)

        total_steps += 1
        return []

    # def update(frame):
    #     global obs, velocities_magnitude, velocities_x, velocities_y
    #     global trajectories, collisions_record, energy_consumption, total_steps
    #
    #     actions = maddpg_agents.choose_action(obs, total_steps, evaluate=True)
    #     obs_, rewards, dones, collision_info = env.step(actions)
    #
    #     for i in range(env.num_agents):
    #         # 记录轨迹
    #         trajectories[i].append(env.multi_current_pos[i])
    #
    #         # 记录碰撞信息
    #         collisions_record[i].append(collision_info)
    #
    #         # 记录速度信息
    #         vel = env.multi_current_vel[i]
    #         v_x, v_y = vel
    #         speed = np.linalg.norm(vel)
    #         velocities_magnitude[i].append(speed)
    #         velocities_x[i].append(v_x)
    #         velocities_y[i].append(v_y)
    #
    #     # 清空当前图像
    #     ax.cla()
    #
    #     # 绘制 UAV 的轨迹
    #     for i in range(env.num_agents):
    #         trajectory = np.array(trajectories[i])
    #         if len(trajectory) > 1:
    #             ax.plot(trajectory[:, 0], trajectory[:, 1], label=f'UAV {i}')
    #
    #     # 绘制 VO 区域
    #     ws_model = env.get_ws_model()
    #     env.VO_Plot(env.multi_current_pos, env.multi_current_vel, ws_model, FLAG=True, ax=ax)
    #
    #     # 绘制 UAV 的当前位置
    #     for i, pos in enumerate(env.multi_current_pos):
    #         ax.scatter(pos[0], pos[1], label=f'UAV {i}', color='blue')
    #
    #     # 绘制障碍物
    #     for obstacle in ws_model['circular_obstacles']:
    #         pos = obstacle['position']
    #         radius = obstacle['radius']
    #         circle = plt.Circle(pos, radius, color='gray', alpha=0.5)
    #         ax.add_patch(circle)
    #
    #     # 设置图像范围和标题
    #     ax.set_xlim(-0.1, env.length + 0.1)
    #     ax.set_ylim(-0.1, env.length + 0.1)
    #     ax.set_title("UAV Simulation with VO Regions")
    #     ax.legend()
    #
    #     # 渲染动画帧
    #     obs = obs_
    #
    #     if any(dones) or frame > 1000:
    #         ani.event_source.stop()
    #         print("Round-up finished in", frame, "steps.")
    #
    #         # 进行评测
    #         evaluator = UAVTaskEvaluator(target_position=[0, 0], max_time_steps=1000, dones=dones)
    #
    #         for i in range(env.num_agents):
    #             result = evaluator.evaluate(trajectories[i], collisions_record[i])
    #             print(f"评测结果 (UAV {i}):")
    #             for key, value in result.items():
    #                 print(f"  {key}: {value:.2f}")
    #
    #         # 平滑数据绘图
    #         smoothed_velocities_magnitude = [moving_average(v, window_size=5) for v in velocities_magnitude]
    #         smoothed_velocities_x = [moving_average(v, window_size=5) for v in velocities_x]
    #         smoothed_velocities_y = [moving_average(v, window_size=5) for v in velocities_y]
    #         time_steps = range(len(smoothed_velocities_magnitude[0]))
    #         plot_velocities(smoothed_velocities_magnitude, smoothed_velocities_x, smoothed_velocities_y)
    #
    #     total_steps += 1
    #     return []


    total_steps = 0

    fig = plt.figure()
    # fig, ax = plt.subplots(figsize=(6, 6))
    ani = animation.FuncAnimation(fig, update, frames=10000, interval=20)
    plt.show()