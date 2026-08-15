import argparse
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from model.maddpg_att import MADDPGWithAttention
from env.sim_env_rvo import UAVEnv
from UAVTask import UAVTaskEvaluator

warnings.filterwarnings('ignore')

RUN_CONFIG_FILENAME = "_last_arvp_rrt2d_run.json"


def load_run_config():
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), RUN_CONFIG_FILENAME)
    if not os.path.exists(cfg_path):
        print(f"[eval_rrt2d][WARN] 未找到配置文件: {cfg_path}，使用默认参数")
        return None
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    print(f"[eval_rrt2d] 从配置加载: {cfg_path}")
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Evaluate ARVP RRT* 2D Augmented Model")
    parser.add_argument('--headless', action='store_true', help='无 GUI 模式，不弹窗口，跑完自动退出')
    parser.add_argument('--from-config', action='store_true', help=f'从 {RUN_CONFIG_FILENAME} 读取路径/维度参数')
    parser.add_argument('--max-frames', type=int, default=200, help='最多评估多少步后退出（headless 模式）')
    parser.add_argument('--episodes', type=int, default=3, help='评估回合数')
    args = parser.parse_args()

    cfg = load_run_config() if args.from_config else None

    if cfg:
        chkpt_dir_with_sep = cfg.get("chkpt_dir_with_sep", "tmp_avoid_dynamic/maddpgwithatt_rrt2D/")
        scenario_name = cfg.get("scenario_name", "UAV_Round_up")
        obs_agt_dim = cfg.get("obs_agt_dim", 38)
        obs_tar_dim = cfg.get("obs_tar_dim", 32)
    else:
        chkpt_dir_with_sep = "tmp_avoid_dynamic/maddpgwithatt_rrt2D/"
        scenario_name = "UAV_Round_up"
        obs_agt_dim = 38
        obs_tar_dim = 32

    env = UAVEnv()
    n_agents = env.num_agents
    n_actions = 2
    actor_dims = []
    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)

    maddpg_agents = MADDPGWithAttention(
        actor_dims, critic_dims, n_agents, n_actions,
        alpha=0.00001, beta=0.001,
        scenario=scenario_name,
        chkpt_dir=chkpt_dir_with_sep,
        obs_agt=obs_agt_dim, obs_tar=obs_tar_dim,
    )

    try:
        maddpg_agents.load_checkpoint()
        print(f"[eval_rrt2d][OK] 模型加载成功: scenario={scenario_name}, dir={chkpt_dir_with_sep}")
    except Exception as e:
        print(f"[eval_rrt2d][WARN] 模型加载失败（可能还未保存）: {type(e).__name__}: {e}")
        if args.headless:
            print("[eval_rrt2d] headless 模式下跳过评估，直接退出 (exit 0)")
            sys.exit(0)
        return

    print('---- Evaluating (RRT* 2D Augmented) ----')

    all_success = []
    all_steps = []
    all_rewards = []

    for ep_idx in range(args.episodes):
        obs = env.reset()
        trajectories = [[] for _ in range(n_agents)]
        collisions_record = []
        episode_reward = 0.0
        episode_step = 0
        dones = [False] * n_agents
        last_mul_pos = None

        while not any(dones):
            actions = maddpg_agents.choose_action(obs, total_steps=0, evaluate=True)
            obs_, rewards, dones, collision_info, mul_pos = env.step(actions)
            last_mul_pos = mul_pos

            for i in range(n_agents):
                trajectories[i].append(np.array(env.multi_current_pos[i]))
            collisions_record.append(collision_info)

            episode_reward += sum(rewards[0:3])
            obs = obs_
            episode_step += 1

            if args.headless and episode_step >= args.max_frames:
                dones = [True] * n_agents
                break

        evaluator = UAVTaskEvaluator([0, 0, 0], max_time_steps=args.max_frames, dones=dones)
        success_eval, total_counts = evaluator.success_evaluate(dones, ep_idx)

        captured = bool(success_eval and episode_step < args.max_frames)
        all_success.append(captured)
        all_steps.append(episode_step)
        all_rewards.append(episode_reward)

        status = "✓ 捕获成功" if captured else "✗ 未捕获"
        print(f"  回合 {ep_idx}: {status} | steps={episode_step} | hunter_total_reward={episode_reward:.2f}")

        if last_mul_pos is not None:
            for i in range(min(n_agents, len(trajectories))):
                try:
                    result = evaluator.evaluate(trajectories[i], collisions_record, last_mul_pos)
                    parts = [f"UAV{i}: "]
                    for k, v in result.items():
                        parts.append(f"{k}={v:.2f}")
                    print("    " + " | ".join(parts))
                except Exception:
                    pass

    n_eval = max(len(all_success), 1)
    success_rate = sum(all_success) / n_eval * 100.0
    avg_steps = np.mean(all_steps) if all_steps else 0.0
    avg_reward = np.mean(all_rewards) if all_rewards else 0.0

    print("=" * 50)
    print(f"[eval_rrt2d][总结] 评估 {n_eval} 回合:")
    print(f"  捕获成功率: {success_rate:.2f}%")
    print(f"  平均步数:   {avg_steps:.1f}")
    print(f"  平均奖励:   {avg_reward:.2f}")
    print("=" * 50)


if __name__ == '__main__':
    main()
