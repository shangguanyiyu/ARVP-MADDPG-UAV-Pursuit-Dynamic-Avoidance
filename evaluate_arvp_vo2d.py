"""
evaluate_arvp_vo2d.py - 方案三 VO2D 的评估脚本
支持:
  --from-config : 从 _last_arvp_vo2d_run.json 读取路径参数 (训练脚本中心化保存)
  --headless    : 无 GUI 模式，跑完即退出 (子进程调用必备)
  --max-frames N: headless 模式下最多跑 N frame (默认 110)
  --no-headless : 强制弹 GUI (调试用)
  --chkpt-dir, --scenario, --stage 等参数可手动覆盖 config
"""
from model.maddpg_att import MADDPGWithAttention
from env.sim_env_vo2d import UAVEnv
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')  # 默认 Agg, 只有真正需要 show 时切 TkAgg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings
import os
import time
import argparse
import json
from UAVTask import UAVTaskEvaluator

warnings.filterwarnings('ignore')

RUN_CONFIG_FILENAME = "_last_arvp_vo2d_run.json"


def load_run_config():
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), RUN_CONFIG_FILENAME)
    if not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[eval_vo2d] 读取 config 失败 {cfg_path}: {e}")
        return None


def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def plot_velocities(velocities_magnitude, velocities_x, velocities_y, save_dir=None):
    time_steps = range(len(velocities_magnitude[0]))
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    for i in range(len(velocities_magnitude)):
        lbl = f'UAV {i}' if i != 3 else 'Target'
        axs[0].plot(time_steps, velocities_magnitude[i], label=lbl)
    axs[0].set_title('Speed Magnitude vs Time (VO2D)')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Speed Magnitude')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    for i in range(len(velocities_x)):
        lbl = f'UAV {i}' if i != 3 else 'Target'
        axs[1].plot(time_steps, velocities_x[i], label=lbl)
    axs[1].set_title('Velocity X Component vs Time (VO2D)')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Velocity X Component')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    for i in range(len(velocities_y)):
        lbl = f'UAV {i}' if i != 3 else 'Target'
        axs[2].plot(time_steps, velocities_y[i], label=lbl)
    axs[2].set_title('Velocity Y Component vs Time (VO2D)')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Velocity Y Component')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, 'vo2d_velocity_plot.png')
        plt.savefig(out, dpi=120)
        print(f"[eval_vo2d] 速度曲线已保存: {out}")
    plt.close(fig)


def run_evaluation(args):
    # ========== 1) 优先从 JSON config 读取 ==========
    cfg = None
    if args.from_config:
        cfg = load_run_config()
        if cfg:
            print(f"[eval_vo2d] 从 {RUN_CONFIG_FILENAME} 读取参数")
            print(f"[eval_vo2d]   method: {cfg.get('method', 'N/A')}")
            print(f"[eval_vo2d]   run_prefix: {cfg.get('run_prefix', 'N/A')}")
            print(f"[eval_vo2d]   model_save_dir: {cfg.get('model_save_dir', 'N/A')}")

    # 命令行参数优先级 > config > 默认
    def resolve(key, default, arg_val=None, cfg_key=None, cfg_map=None):
        if arg_val is not None and (isinstance(arg_val, str) and arg_val != "" or not isinstance(arg_val, str)):
            # argparse 提供的非空值
            return arg_val
        if cfg is not None:
            ck = cfg_key or key
            if cfg_map:
                return cfg_map(cfg.get(ck, default))
            return cfg.get(ck, default)
        return default

    obs_agt_dim = resolve('obs_agt_dim', 32, arg_val=args.obs_agt, cfg_key='obs_agt_dim')
    obs_tar_dim = resolve('obs_tar_dim', 29, arg_val=args.obs_tar, cfg_key='obs_tar_dim')
    n_agents_cfg = resolve('n_agents', 4, cfg_key='n_agents')
    n_actions_cfg = resolve('n_actions', 2, cfg_key='n_actions')
    scenario_name = resolve('scenario_name', 'UAV_Round_up', arg_val=args.scenario, cfg_key='scenario_name')
    chkpt_dir_with_sep = resolve('chkpt_dir_with_sep', 'tmp_avoid_dynamic/maddpgwithatt_vo2d/',
                                 arg_val=args.chkpt_dir, cfg_key='chkpt_dir_with_sep')
    model_save_dir = resolve('model_save_dir', None, cfg_key='model_save_dir')
    log_dir = resolve('log_dir', None, cfg_key='log_dir')

    # 若 chkpt_dir 未显式以 / 结尾，确保结尾
    if not chkpt_dir_with_sep.endswith(os.sep):
        chkpt_dir_with_sep = chkpt_dir_with_sep + os.sep
    print(f"[eval_vo2d] chkpt_dir = {chkpt_dir_with_sep}")
    print(f"[eval_vo2d] scenario  = {scenario_name}")
    print(f"[eval_vo2d] obs dim: agt={obs_agt_dim}, tar={obs_tar_dim}")

    # ========== 2) 非 headless 时切换 GUI backend ==========
    headless = args.headless
    max_frames = args.max_frames if args.max_frames is not None else 110
    if not headless:
        try:
            matplotlib.use('TkAgg', force=True)
            import matplotlib.pyplot as plt_gui
        except Exception as e:
            print(f"[eval_vo2d] 无法启用 TkAgg GUI，回退 Agg headless: {e}")
            headless = True

    env = UAVEnv()
    n_agents = env.num_agents
    n_actions = 2
    actor_dims = []
    velocities_magnitude = [[] for _ in range(env.num_agents)]
    velocities_x = [[] for _ in range(env.num_agents)]
    velocities_y = [[] for _ in range(env.num_agents)]
    trajectories = [[] for _ in range(env.num_agents)]
    collisions_record = [[] for _ in range(env.num_agents)]

    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)

    maddpg_agents = MADDPGWithAttention(actor_dims, critic_dims, n_agents, n_actions,
                                        alpha=0.00001, beta=0.001,
                                        scenario=scenario_name,
                                        chkpt_dir=chkpt_dir_with_sep,
                                        obs_agt=obs_agt_dim, obs_tar=obs_tar_dim)

    maddpg_agents.load_checkpoint()
    print('---- Evaluating (VO2D) ----')

    obs = env.reset()
    total_steps = 0
    finish_frame = None
    final_mul_pos = None

    # ========== 3) headless: 直接跑 loop, 不创建动画 ==========
    if headless:
        for frame in range(max_frames):
            actions = maddpg_agents.choose_action(obs, total_steps, evaluate=True)
            obs_, rewards, dones, collision_info, mul_pos = env.step(actions)

            for i in range(env.num_agents):
                trajectories[i].append(env.multi_current_pos[i].copy())
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
                finish_frame = frame
                final_mul_pos = mul_pos
                print(f"[eval_vo2d] Round-up finished in {frame} steps.")
                break

        if finish_frame is None:
            finish_frame = max_frames - 1
            final_mul_pos = env.multi_current_pos
            print(f"[eval_vo2d] 达到 max-frames={max_frames}，停止评估。")

        # ===== 评测 =====
        evaluator = UAVTaskEvaluator([0, 0, 0], max_time_steps=max_frames,
                                     dones=[any(dones)] * env.num_agents)
        eval_results_all = {}
        for i in range(env.num_agents):
            result = evaluator.evaluate(trajectories[i], collisions_record[i], final_mul_pos)
            eval_results_all[i] = result
            print(f"评测结果 (UAV {i}):")
            for key, value in result.items():
                print(f"  {key}: {value:.2f}")

        # 保存速度图 & 评测结果
        save_fig_dir = log_dir or model_save_dir or os.path.join('log_arvp_vo2d', 'eval_last')
        plot_velocities(velocities_magnitude, velocities_x, velocities_y, save_dir=save_fig_dir)

        # 保存 JSON 摘要
        if save_fig_dir:
            os.makedirs(save_fig_dir, exist_ok=True)
            summary_path = os.path.join(save_fig_dir, 'vo2d_eval_summary.json')
            summary = {
                "finished_frame": finish_frame,
                "evaluator_results": {
                    k: {kk: float(vv) for kk, vv in v.items()}
                    for k, v in eval_results_all.items()
                },
                "method": cfg.get('method') if cfg else '方案三-VO2D',
                "run_prefix": cfg.get('run_prefix') if cfg else None,
            }
            try:
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                print(f"[eval_vo2d] 评测摘要保存: {summary_path}")
            except Exception as e:
                print(f"[eval_vo2d] 摘要保存失败: {e}")

        env.close()
        return

    # ========== 4) GUI: 用 FuncAnimation ==========
    fig = plt.figure(figsize=(8, 8))

    def update(frame):
        nonlocal obs, velocities_magnitude, velocities_x, velocities_y
        nonlocal trajectories, collisions_record, total_steps, finish_frame, final_mul_pos

        actions = maddpg_agents.choose_action(obs, total_steps, evaluate=True)
        obs_, rewards, dones, collision_info, mul_pos = env.step(actions)

        for i in range(env.num_agents):
            trajectories[i].append(env.multi_current_pos[i].copy())
            collisions_record[i].append(collision_info)
            vel = env.multi_current_vel[i]
            v_x, v_y = vel
            speed = np.linalg.norm(vel)
            velocities_magnitude[i].append(speed)
            velocities_x[i].append(v_x)
            velocities_y[i].append(v_y)

        env.render_anime(frame)
        obs = obs_

        if any(dones) or frame > max_frames:
            ani.event_source.stop()
            finish_frame = frame
            final_mul_pos = mul_pos
            print(f"Round-up finished in {frame} steps.")

            evaluator = UAVTaskEvaluator([0, 0, 0], max_time_steps=max_frames,
                                         dones=[any(dones)] * env.num_agents)
            for i in range(env.num_agents):
                result = evaluator.evaluate(trajectories[i], collisions_record[i], mul_pos)
                print(f"评测结果 (UAV {i}):")
                for key, value in result.items():
                    print(f"  {key}: {value:.2f}")

            smoothed_vm = [moving_average(v, window_size=5) for v in velocities_magnitude]
            smoothed_vx = [moving_average(v, window_size=5) for v in velocities_x]
            smoothed_vy = [moving_average(v, window_size=5) for v in velocities_y]
            save_dir = log_dir or model_save_dir
            plot_velocities(smoothed_vm, smoothed_vx, smoothed_vy, save_dir=save_dir)

        total_steps += 1
        return []

    ani = animation.FuncAnimation(fig, update, frames=max_frames + 20, interval=50)
    plt.show()
    env.close()


def main():
    parser = argparse.ArgumentParser(description="ARVP 方案三 VO2D 评估脚本")
    parser.add_argument('--from-config', action='store_true', default=False,
                        help=f'从 {RUN_CONFIG_FILENAME} 读取参数（推荐）')
    parser.add_argument('--chkpt-dir', type=str, default=None,
                        help='模型 checkpoint 根目录 (结尾加分隔符)')
    parser.add_argument('--scenario', type=str, default=None,
                        help='场景名 (即 chkpt 子目录名)')
    parser.add_argument('--stage', type=str, default=None, help='stage (仅日志)')
    parser.add_argument('--obs-agt', type=int, default=None, help='hunter 观测维度 (默认32)')
    parser.add_argument('--obs-tar', type=int, default=None, help='target 观测维度 (默认29)')
    parser.add_argument('--headless', action='store_true', default=False,
                        help='无 GUI 模式，跑完即退出 (训练子进程调用)')
    parser.add_argument('--no-headless', dest='headless', action='store_false',
                        help='强制启用 GUI 模式')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='headless 模式最多跑多少步 (默认 110)')
    args = parser.parse_args()

    run_evaluation(args)


if __name__ == '__main__':
    main()
