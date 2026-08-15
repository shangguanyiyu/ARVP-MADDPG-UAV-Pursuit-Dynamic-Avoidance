"""
evaluate_arvp.py
================
评估脚本。

路径中心化说明：
  本脚本不再硬编码任何模型目录，所有路径/维度参数都从 train_arvp2D.py 写出的
  _last_arvp2D_run.json 读取（通过 --from-config），或通过命令行参数显式传入。
  这样 train_arvp2D.py 子进程调用本脚本时，可以加载到正确的模型检查点路径。

用法：
  # 训练时由 train_arvp2D.py 自动调用（无 GUI，跑完即退出）：
  python evaluate_arvp.py --headless --from-config --max-frames 50

  # 也可以手动指定路径（覆盖配置文件）：
  python evaluate_arvp.py --headless \
      --chkpt-dir tmp_avoid_dynamic/maddpgwithatt_2D/ \
      --scenario 0815_103020_stage1_UAV_Round_up \
      --obs-agt 26 --obs-tar 23 --max-frames 100

  # 带动画的可视化评估（需要图形环境）：
  python evaluate_arvp.py --from-config
"""
from model.maddpg_att import MADDPGWithAttention
from env.sim_env_rvo import UAVEnv
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')  # 无显示环境也能 import / 保存图
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings
import os
import time
import argparse
import json
from UAVTask import UAVTaskEvaluator

warnings.filterwarnings('ignore')

# 默认配置文件名（与 train_arvp2D.py 保持一致），位于本脚本同目录
DEFAULT_CONFIG_FILENAME = "_last_arvp2D_run.json"


def load_run_config(config_path: str) -> dict:
    """读取 train_arvp2D.py 写出的运行配置 JSON"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resolve_config_path(args) -> str:
    """定位配置 JSON 文件路径"""
    if args.config and os.path.isfile(args.config):
        return args.config
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, DEFAULT_CONFIG_FILENAME)
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(
        f"未找到配置文件 {DEFAULT_CONFIG_FILENAME}（位于 {here}）。"
        f"请先运行 train_arvp2D.py，或通过 --config / --chkpt-dir 显式指定路径。")


def build_eval_params(args):
    """合并配置文件与命令行参数，命令行参数优先级更高。返回评估所需路径/维度字典。"""
    cfg = {}
    if args.from_config or (not args.chkpt_dir and not args.scenario):
        cfg_path = resolve_config_path(args)
        cfg = load_run_config(cfg_path)
        print(f"[evaluate] 读取配置: {cfg_path}")

    chkpt_dir = args.chkpt_dir or cfg.get("chkpt_dir_with_sep", "tmp_avoid_dynamic/maddpgwithatt_2D/")
    scenario = args.scenario or cfg.get("scenario_name", "UAV_Round_up")
    obs_agt = args.obs_agt if args.obs_agt is not None else cfg.get("obs_agt_dim", 26)
    obs_tar = args.obs_tar if args.obs_tar is not None else cfg.get("obs_tar_dim", 23)

    model_save_dir = os.path.join(chkpt_dir.rstrip(os.sep), scenario)
    print(f"[evaluate] chkpt_dir = {chkpt_dir}")
    print(f"[evaluate] scenario  = {scenario}")
    print(f"[evaluate] 模型目录  = {model_save_dir}")
    print(f"[evaluate] obs_agt={obs_agt}, obs_tar={obs_tar}")

    if not os.path.isdir(model_save_dir):
        print(f"[evaluate][警告] 模型目录不存在: {model_save_dir}")

    return {
        "chkpt_dir": chkpt_dir,
        "scenario": scenario,
        "obs_agt": obs_agt,
        "obs_tar": obs_tar,
        "model_save_dir": model_save_dir,
    }


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


def run_headless(env, maddpg_agents, max_frames=50):
    """无 GUI 的快速评估：最多跑 max_frames 步，结束后打印每个 UAV 的评测结果。"""
    n_agents = env.num_agents
    velocities_magnitude = [[] for _ in range(n_agents)]
    velocities_x = [[] for _ in range(n_agents)]
    velocities_y = [[] for _ in range(n_agents)]
    trajectories = [[] for _ in range(n_agents)]
    collisions_record = [[] for _ in range(n_agents)]

    obs = env.reset()
    total_steps = 0
    mul_pos = None
    dones = [False] * n_agents

    print(f"[evaluate][headless] 开始评估，最多 {max_frames} 步...")
    while total_steps < max_frames:
        actions = maddpg_agents.choose_action(obs, total_steps, evaluate=True)
        obs_, rewards, dones, collision_info, mul_pos = env.step(actions)

        for i in range(n_agents):
            trajectories[i].append(env.multi_current_pos[i])
            collisions_record[i].append(collision_info)
            vel = env.multi_current_vel[i]
            v_x, v_y = vel
            speed = np.linalg.norm(vel)
            velocities_magnitude[i].append(speed)
            velocities_x[i].append(v_x)
            velocities_y[i].append(v_y)

        obs = obs_
        total_steps += 1

        if any(dones):
            print(f"[evaluate][headless] Round-up finished in {total_steps} steps.")
            break

    if total_steps >= max_frames and not any(dones):
        print(f"[evaluate][headless] 达到最大步数 {max_frames}，未捕获到 done。")

    # 进行评测
    evaluator = UAVTaskEvaluator([0, 0, 0], max_time_steps=max(max_frames, 1), dones=dones)

    print("========== 评测结果 ==========")
    any_result = False
    for i in range(n_agents):
        try:
            result = evaluator.evaluate(trajectories[i], collisions_record[i], mul_pos)
            any_result = True
            print(f"评测结果 (UAV {i}):")
            for key, value in result.items():
                if isinstance(value, (int, float, np.floating)):
                    print(f"  {key}: {float(value):.2f}")
                else:
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"[evaluate][headless] UAV {i} 评测失败: {type(e).__name__}: {e}")

    if not any_result:
        print("[evaluate][headless] 无可用评测结果。")

    print(f"[evaluate][headless] 评估结束，共 {total_steps} 步。")
    return total_steps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ARVP 评估脚本（路径参数化）")
    parser.add_argument('--from-config', action='store_true',
                        help='优先从 _last_arvp2D_run.json 读取路径/维度参数')
    parser.add_argument('--config', type=str, default=None,
                        help='指定配置 JSON 文件路径（默认使用脚本同目录的 _last_arvp2D_run.json）')
    parser.add_argument('--chkpt-dir', type=str, default=None,
                        help='模型检查点根目录（末尾带分隔符），覆盖配置文件')
    parser.add_argument('--scenario', type=str, default=None,
                        help='场景名（会拼接到 chkpt-dir 后），覆盖配置文件')
    parser.add_argument('--obs-agt', type=int, default=None,
                        help='agent 观测维度，覆盖配置文件')
    parser.add_argument('--obs-tar', type=int, default=None,
                        help='target 观测维度，覆盖配置文件')
    parser.add_argument('--headless', action='store_true',
                        help='无 GUI 模式：跑 --max-frames 步后自动退出')
    parser.add_argument('--max-frames', type=int, default=1000,
                        help='headless 模式下的最大步数（默认 1000）')
    args = parser.parse_args()

    params = build_eval_params(args)

    env = UAVEnv()
    n_agents = env.num_agents
    n_actions = 2
    actor_dims = []
    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)

    maddpg_agents = MADDPGWithAttention(actor_dims, critic_dims, n_agents, n_actions,
                                        alpha=0.00001, beta=0.00001,
                                        scenario=params["scenario"],
                                        chkpt_dir=params["chkpt_dir"],
                                        obs_agt=params["obs_agt"],
                                        obs_tar=params["obs_tar"])

    maddpg_agents.load_checkpoint()
    print('---- Evaluating ----')

    # ===================== headless 模式：跑完即退出 =====================
    if args.headless:
        run_headless(env, maddpg_agents, max_frames=args.max_frames)
        sys.exit(0)

    # ===================== 可视化动画模式（需要图形环境） =====================
    velocities_magnitude = [[] for _ in range(env.num_agents)]
    velocities_x = [[] for _ in range(env.num_agents)]
    velocities_y = [[] for _ in range(env.num_agents)]
    trajectories = [[] for _ in range(env.num_agents)]
    collisions_record = [[] for _ in range(env.num_agents)]
    energy_consumption = [0 for _ in range(env.num_agents)]

    obs = env.reset()


    def update(frame):
        global obs, velocities_magnitude, velocities_x, velocities_y
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
            v_x, v_y = vel
            speed = np.linalg.norm(vel)
            velocities_magnitude[i].append(speed)
            velocities_x[i].append(v_x)
            velocities_y[i].append(v_y)

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
            time_steps = range(len(smoothed_velocities_magnitude[0]))
            plot_velocities(smoothed_velocities_magnitude, smoothed_velocities_x, smoothed_velocities_y)

        total_steps += 1
        return []


    total_steps = 0

    fig = plt.figure()
    ani = animation.FuncAnimation(fig, update, frames=10000, interval=20)
    plt.show()
