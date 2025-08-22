from maddpg import MADDPG
from sim_env import UAVEnv
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings
import os
import time
from UAVTask import UAVTaskEvaluator
from sim_env import obstacle

warnings.filterwarnings('ignore')


def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def plot_velocity_magnitude(time_steps, velocities_magnitude):
    plt.figure(figsize=(15, 4))
    for i in range(len(velocities_magnitude)):
        if i != 3:
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
        if i != 3:
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
        if i != 3:
            plt.plot(time_steps, velocities_y[i], label=f'UAV {i}')
        else:
            plt.plot(time_steps, velocities_y[i], label='Target')
    plt.xlabel("Time Steps")
    plt.ylabel("$vel_y$")
    plt.title("UAV $Vel_y$")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_velocities(velocities_magnitude, velocities_x, velocities_y):
    time_steps = range(len(velocities_magnitude[0]))
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

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

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    MAX_STEPS = 300
    EVALUATION_TIMES = 50  # 定义评测次数
    env = UAVEnv()
    n_agents = env.num_agents
    n_actions = 2
    actor_dims = []
    velocities_magnitude = [[] for _ in range(env.num_agents)]  # 记录速度大小
    velocities_x = [[] for _ in range(env.num_agents)]  # 记录 x 方向速度
    velocities_y = [[] for _ in range(env.num_agents)]  # 记录 y 方向速度


    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions,
                           fc1=128, fc2=128, alpha=0.0001, beta=0.003, scenario='UAV_Round_up',
                           chkpt_dir='tmp_back1/maddpg/')

    maddpg_agents.load_checkpoint()
    print('---- Evaluating ----')
    # obstacles = [obstacle(length=2) for _ in range(env.num_obstacle)]  # 假设环境中有
    obstacles = [{"position": obs.position, "radius": obs.radius} for obs in
                 [obstacle(length=2) for _ in range(env.num_obstacle)]] # 位置 + 半径 字典
    evaluator = UAVTaskEvaluator(target_position=[], max_time_steps=MAX_STEPS)  # 初始化评测类

    results = []  # 用于存储每次评测的结果
    steps_per_evaluation = []  # 用于记录每次评测的步长

    for times in range(1, EVALUATION_TIMES + 1):  # 循环 EVALUATION_TIMES 次
        obs = env.reset()  # 重置环境
        total_steps = 0
        done = False
        current_pos = [[] for _ in range(env.num_agents - 1)]
        tar_pos = []
        obs_pos = []
        obs_radius = []

        while not done and total_steps < MAX_STEPS:
            # 记录速度信息
            for i in range(env.num_agents):
                vel = env.multi_current_vel[i]
                v_x, v_y = vel
                speed = np.linalg.norm(vel)

                velocities_magnitude[i].append(speed)
                velocities_x[i].append(v_x)
                velocities_y[i].append(v_y)
            for i in range(env.num_agents - 1):
                current_pos[i].append(env.multi_current_pos[i]) # agent
            tar_pos.append(env.multi_current_pos) # tar
            # # 更新并记录障碍物位置
            # current_obs_pos = []
            # for obs_obj in obstacles:
            #     current_obs_pos.append(obs_obj.position.copy())  # 记录当前障碍物位置
            #     # obs_obj.position += obs_obj.velocity  # 更新障碍物位置
            #     # # 检查边界条件，防止障碍物超出范围
            #     # if np.any(obs_obj.position < 0.45) or np.any(obs_obj.position > 2 - 0.55):
            #     #     obs_obj.velocity = -obs_obj.velocity  # 反向移动
            # obs_pos.append(current_obs_pos)  # 记录所有障碍物的位置
            # 更新并记录障碍物位置
            current_obs_pos = []
            current_obs_radius = []
            for obs_obj in obstacles:
                current_obs_pos.append(obs_obj["position"].copy())  # 记录当前障碍物位置
                current_obs_radius.append(obs_obj["radius"])
            obs_pos.append(current_obs_pos)  # 记录所有障碍物的位置
            obs_radius = current_obs_radius

            # 选择动作并执行
            actions = maddpg_agents.choose_action(obs, total_steps, evaluate=True)
            obs_, _, dones = env.step(actions)
            obs = obs_
            total_steps += 1

            # 获取目标位置
            done = all(dones)
        # 进行评测
        # print(obs_radius)
        evaluate_results = evaluator.evaluate(current_pos, obs_pos, obs_radius, tar_pos, done, total_steps)
        if evaluate_results is not None:
            results.append(evaluate_results)  # 存储评测结果
        # 记录此次评测的步长
        steps_per_evaluation.append(total_steps)
        print(f"第 {times} 次评测完成，步长为：{total_steps}")

        # 在每轮结束后清零 success_count
        evaluator.success_count = 0
        evaluator.collision_count = 0  # 碰撞次数

    # 打印所有评测的统计信息
    print("\n评测完成！所有评测结果如下：")
    for i, res in enumerate(results, 1):
        print(f"第{i}次评测结果：", res)

    average_calcualtion = evaluator.calculate_metrics(EVALUATION_TIMES, results)
    # "arrival_rate": arrival_rate,
    # "average_collision": average_collision
    # 打印结果
    print("\n总体评测统计结果：")
    print(f"  到达率：{average_calcualtion['arrival_rate']:.2f}")
    print(f"  平均碰撞次数：{average_calcualtion['average_collision']:.2f}")


    # smoothed_velocities_magnitude = [moving_average(vel, window_size=5) for vel in velocities_magnitude]
    # smoothed_velocities_x = [moving_average(vel, window_size=5) for vel in velocities_x]
    # smoothed_velocities_y = [moving_average(vel, window_size=5) for vel in velocities_y]
    #
    # time_steps = range(len(smoothed_velocities_magnitude[0]))
    # plot_velocity_magnitude(time_steps, smoothed_velocities_magnitude)
    # plot_velocity_x(time_steps, smoothed_velocities_x)
    # plot_velocity_y(time_steps, smoothed_velocities_y)
