"""
train_arvp_rrt2D.py — ARVP + RRT动态取样（NoStage1）训练脚本
特性：
  * mu3=0.4：跳过第一阶段，直接开启围捕（encircle/capture）奖励
  * RRT每步重规划 + 锚点120°均匀分配 + 短截动态取样参考点
  * r_rrt对齐奖励注入奖励函数，引导策略沿RRT参考方向运动
  * 每200个episode打印训练状态（奖励、成功率、RRT规划统计等）
  * 总回合数 N_GAMES = 5000
"""
import numpy as np
from model.maddpg_att import MADDPGWithAttention
from env.sim_env_rvo import UAVEnv
from buffer.PERbuffer import PERMultiAgentReplayBuffer
import time
import os
import matplotlib.pyplot as plt
import warnings
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import subprocess
from UAVTask import UAVTaskEvaluator

warnings.filterwarnings('ignore')


def create_unique_log_dir(base_dir="runs/UAV_training"):
    idx = 1
    while True:
        log_dir = f"{base_dir}{idx:02d}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            return log_dir
        idx += 1


def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def obs_list_to_state_vector(obs):
    return np.hstack([np.ravel(o) for o in obs])


if __name__ == '__main__':

    env = UAVEnv()
    log_dir = create_unique_log_dir()
    writer = SummaryWriter(log_dir=log_dir)

    n_agents = env.num_agents
    actor_dims = []
    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)

    n_actions = 2
    alpha = 0.00001
    beta = 0.001
    # NoStage1 专用保存目录（避免与stage1模型冲突）
    chkpt_dir = 'tmp_arvp_rrt2D_nostage1/maddpgwithatt/'
    os.makedirs(chkpt_dir, exist_ok=True)

    maddpg_agents = MADDPGWithAttention(actor_dims, critic_dims, n_agents, n_actions,
                                        alpha=alpha, beta=beta, scenario='UAV_Round_up',
                                        chkpt_dir=chkpt_dir)
    # 保存超参数
    hyperparams_path = os.path.join(chkpt_dir, 'hyperparameters.txt')
    with open(hyperparams_path, 'w') as f:
        f.write(f'Branch: trae/arvp_rrt2DNoStage1 (NoStage1 + RRT2D dynamic sampling)\n')
        f.write(f'mu3: 0.4 (直接围捕，无stage1预训练)\n')
        f.write(f'RRT enabled: {env.rrt_enabled}\n')
        f.write(f'RRT capture radius: {env.rrt_capture_radius}\n')
        f.write(f'RRT look ahead: {env.rrt_look_ahead:.4f}\n')
        f.write(f'alpha: {alpha}\n')
        f.write(f'beta: {beta}\n')
        f.write(f'N_GAMES: 5000\n')

    memory = PERMultiAgentReplayBuffer(1000000, critic_dims, actor_dims,
                                       n_actions, n_agents, batch_size=256)

    BATCH_SIZE = 100
    N_GAMES = 5000
    MAX_STEPS = 110
    PRINT_INTERVAL = 200  # 每200个episode打印训练状态
    # ---------- Resume 模式 ----------
    # 设为 0 表示从0开始；设为 >0 表示从该 episode 继续（需checkpoint存在）
    RESUME_FROM_EP = int(os.environ.get('RESUME_FROM_EP', 0))
    RESUME_MODE = RESUME_FROM_EP > 0
    total_steps = 0
    score_history = []
    target_score_history = []
    evaluate = False
    best_score = -1000
    window_size = 100

    # ---------- 额外统计：RRT规划成功率、围捕完成率等 ----------
    rrt_plan_success_steps = 0   # RRT规划成功的step数
    total_env_steps = 0          # 总step数
    capture_episode_count = 0    # 成功完成围捕（Sum_S==S4且d<=d_capture）的回合数
    capture_steps_records = []   # 成功围捕回合所用的步数

    evaluator = UAVTaskEvaluator([0, 0, 0], max_time_steps=MAX_STEPS,
                                 dones=[False, False, False, False])

    print('============================================================')
    print('  ARVP-RRT2D NoStage1 训练开始')
    print('  Branch      : trae/arvp_rrt2DNoStage1')
    print('  mu3         : 0.4 (直接围捕，跳过stage1)')
    print('  N_GAMES     :', N_GAMES)
    print('  PRINT every :', PRINT_INTERVAL, 'episodes')
    print('  Checkpoint  :', chkpt_dir)
    print('  Log dir     :', log_dir)
    print('  RRT enabled :', env.rrt_enabled)
    if RESUME_MODE:
        print('  RESUME MODE : 从 ep', RESUME_FROM_EP, '继续训练（加载checkpoint）')
        maddpg_agents.load_checkpoint()
        # 从之前checkpoint估算best_score（取最近一次保存的近似值，避免重复保存）
        # 读hyperparameters.txt 中的 best_score（如果有）
        best_score_path = os.path.join(chkpt_dir, 'best_score.txt')
        if os.path.exists(best_score_path):
            with open(best_score_path) as f:
                best_score = float(f.read().strip())
            print(f'  Loaded best_score = {best_score:.2f} from {best_score_path}')
        else:
            # 未记录best_score时给一个较低值，允许首次保存覆盖
            best_score = -1e9
            print(f'  best_score not recorded, using {best_score}')
    print('============================================================')

    i = RESUME_FROM_EP  # resume时从断点继续；否则从0
    while i < N_GAMES:
        with tqdm(total=BATCH_SIZE, desc=f"Batch Progress (Total {i}/{N_GAMES})", unit="episode",
                  disable=True) as pbar:
            for _ in range(BATCH_SIZE):
                if i >= N_GAMES:
                    break

                obs = env.reset()
                score = 0
                score_target = 0
                dones = [False] * n_agents
                episode_step = 0

                while not any(dones):
                    actions = maddpg_agents.choose_action(obs, total_steps, evaluate)
                    obs_, rewards, dones, collision_info, mul_pos = env.step(actions)

                    # ---- 统计RRT规划成功率（参考点非None说明规划有效） ----
                    total_env_steps += 1
                    if all(rp is not None for rp in env.rrt_reference_points):
                        rrt_plan_success_steps += 1

                    success_evaluator_total, total_counts = evaluator.success_evaluate(dones, i)

                    state = obs_list_to_state_vector(obs)
                    state_ = obs_list_to_state_vector(obs_)

                    if episode_step >= MAX_STEPS:
                        dones = [True] * n_agents

                    memory.store_transition(obs, state, actions, rewards, obs_, state_, dones)

                    if total_steps % 10 == 0 and not evaluate:
                        maddpg_agents.learn(memory, total_steps, state)

                    obs = obs_
                    score += sum(rewards[0:3])
                    score_target += rewards[-1]
                    total_steps += 1
                    episode_step += 1

                # ----- episode结束后统计 -----
                # 是否成功围捕：dones全员=True且episode_step < MAX_STEPS
                if all(dones) and episode_step < MAX_STEPS:
                    capture_episode_count += 1
                    capture_steps_records.append(episode_step)
                # 成功率日志
                if success_evaluator_total and episode_step < MAX_STEPS:
                    tqdm.write(
                        f"  [Capture] ep {i:4d} | total success rate: "
                        f"{success_evaluator_total * 100:.2f}% (episode steps={episode_step})"
                    )

                score_history.append(score)
                target_score_history.append(score_target)

                writer.add_scalar("UAV Reward", score, i)
                writer.add_scalar("Target Reward", score_target, i)

                # ----- 每 PRINT_INTERVAL 回合打印详细训练状态 -----
                if (i + 1) % PRINT_INTERVAL == 0:
                    recent_scores = score_history[-PRINT_INTERVAL:]
                    recent_target_scores = target_score_history[-PRINT_INTERVAL:]
                    avg_score = float(np.mean(recent_scores))
                    avg_target = float(np.mean(recent_target_scores))
                    best_recent = float(np.max(recent_scores))
                    worst_recent = float(np.min(recent_scores))
                    # 最近PRINT_INTERVAL回合内的围捕成功率
                    recent_capture_rate = (
                        capture_episode_count -
                        (capture_episode_count - len([s for s in recent_scores if s > 0]))
                        # 近似：直接用 capture_episode_count / (i+1) 累积，
                        # 再减去之前的估计值，更简单是另外维护
                    )
                    # 更可靠的估计：近N回合完成围捕的回合数
                    recent_start = max(0, i + 1 - PRINT_INTERVAL)
                    # 由于capture记录只计数，这里重算近似：
                    # 通过reward峰值是否超过 finish reward threshold (mu4*10*3=300) 来判断
                    finish_threshold = 100.0  # 三机加起来奖励至少超过100
                    recent_capture_count = sum(
                        1 for s in recent_scores if s > finish_threshold
                    )
                    avg_capture_steps = (
                        float(np.mean(capture_steps_records[-recent_capture_count:]))
                        if recent_capture_count > 0 and len(capture_steps_records) > 0
                        else float('nan')
                    )
                    rrt_success_rate = (
                        rrt_plan_success_steps / total_env_steps * 100.0
                        if total_env_steps > 0 else 0.0
                    )
                    cumu_capture_rate = capture_episode_count / (i + 1) * 100.0
                    print()
                    print('=' * 62)
                    print(f'  [Status Report] Episodes {recent_start:4d} ~ {i + 1:4d} / {N_GAMES}')
                    print('-' * 62)
                    print(f'  Avg Reward (hunters)  : {avg_score:8.2f}')
                    print(f'  Avg Reward (target)   : {avg_target:8.2f}')
                    print(f'  Best / Worst (recent) : {best_recent:.2f} / {worst_recent:.2f}')
                    print(f'  Capture count (recent): {recent_capture_count} / {PRINT_INTERVAL}  '
                          f'({recent_capture_count / PRINT_INTERVAL * 100:.1f}%)')
                    if not np.isnan(avg_capture_steps):
                        print(f'  Avg capture steps     : {avg_capture_steps:.1f} steps')
                    print(f'  Capture cumulative    : {capture_episode_count} / {i + 1}  '
                          f'({cumu_capture_rate:.1f}%)')
                    print(f'  RRT plan success rate : {rrt_success_rate:.1f}%  '
                          f'(steps: {rrt_plan_success_steps}/{total_env_steps})')
                    print(f'  Total env steps       : {total_env_steps}')
                    print(f'  Best score so far     : {best_score:.2f}')
                    print('=' * 62)
                    # 同步到TensorBoard
                    writer.add_scalar("Avg Reward (last 200)", avg_score, i)
                    writer.add_scalar("Capture Rate (last 200)",
                                      recent_capture_count / PRINT_INTERVAL, i)
                    writer.add_scalar("Cumulative Capture Rate", cumu_capture_rate / 100.0, i)
                    writer.add_scalar("RRT Plan Success Rate", rrt_success_rate / 100.0, i)

                i += 1

            # ---------- 每一个 BATCH 结束，计算并保存模型 ----------
            avg_score = float(np.mean(score_history[-BATCH_SIZE:])) \
                if len(score_history) >= BATCH_SIZE else float(np.mean(score_history))
            avg_target_score = float(np.mean(target_score_history[-BATCH_SIZE:])) \
                if len(target_score_history) >= BATCH_SIZE else float(np.mean(target_score_history))

            tqdm.write(f'Batch {i // BATCH_SIZE} done.  '
                       f'Avg Score={avg_score:.2f}  Avg Target={avg_target_score:.2f}')
            tqdm.write(f'Model checkpoint candidate at batch {i // BATCH_SIZE}.')

            if avg_score > best_score:
                tqdm.write(f'  -> New best score {avg_score:.2f} > prev {best_score:.2f}, '
                           f'saving models & running quick evaluate...')
                maddpg_agents.save_checkpoint()
                best_score = avg_score
                # 保存 best_score 以便 resume 时恢复
                with open(os.path.join(chkpt_dir, 'best_score.txt'), 'w') as f:
                    f.write(f'{best_score:.6f}')
                # 评估10秒（与原训练脚本保持一致）
                process = subprocess.Popen(['python', 'evaluate_arvp.py'])
                time.sleep(10)
                process.terminate()
                process.wait()

    # ---------- 训练结束：绘制奖励曲线 ----------
    score_history_ma = moving_average(score_history, window_size) \
        if len(score_history) >= window_size else None
    target_score_history_ma = moving_average(target_score_history, window_size) \
        if len(target_score_history) >= window_size else None

    plt.figure(figsize=(12, 6))
    plt.plot(score_history, label='Raw Total Reward', alpha=0.3)
    if score_history_ma is not None:
        plt.plot(range(window_size - 1, len(score_history)),
                 score_history_ma, label='Smoothed Total Reward')
    plt.title(f'ARVP-RRT2D NoStage1 Reward ({N_GAMES} episodes, mu3=0.4)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('reward_curves_rrt2D_nostage1.png')
    print(f'\nReward curve saved to reward_curves_rrt2D_nostage1.png')

    # ---------- 最终汇总打印 ----------
    print()
    print('============================================================')
    print('  ARVP-RRT2D NoStage1 训练完成！')
    print(f'  Total episodes        : {i}')
    print(f'  Total env steps       : {total_env_steps}')
    print(f'  Cumulative captures   : {capture_episode_count}')
    print(f'  Capture rate          : {capture_episode_count / i * 100:.1f}%')
    if capture_steps_records:
        print(f'  Avg capture steps     : {np.mean(capture_steps_records):.1f}')
    rrt_success_rate = rrt_plan_success_steps / total_env_steps * 100.0 if total_env_steps else 0.0
    print(f'  RRT success rate      : {rrt_success_rate:.1f}%')
    print(f'  Best score            : {best_score:.2f}')
    print(f'  Checkpoint dir        : {chkpt_dir}')
    print(f'  TensorBoard log       : {log_dir}')
    print('============================================================')

    writer.close()
