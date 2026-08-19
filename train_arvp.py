"""
train_arvp.py (trae/arvpNoise 改造版)
======================================
在原 ARVP + Attention 训练脚本基础上,以最小改动接入声呐信号模拟:
  1) env/sim_env_rvo.py 中 hunter 的 obs 已从 26 扩到 42 (26 原 + 16 声呐时域采样)
  2) 模型换成 MADDPGNoiseWithAttention → 其内部分支:
       - HunterActorNetwork: base_obs(26) FC + 信号分支 SignalEncoder(FFT+GRU+Attention) + 融合
       - CriticNetwork_X:  原 ATT attention 上下文 + 3 个 hunter 的 SignalEncoder 输出融合
  3) 不修改 reward 机制 (用户明确要求)
  4) 修复了: TensorBoard 异步写文件丢失导致崩溃、checkpoint 保存路径未校验、
            SummaryWriter 用绝对路径 + 容错 try/except

每 100 个 episode (1 个 batch) 打印 avg_score / avg_target / 成功率 / 步数。
"""
import argparse
import numpy as np
from sympy import evaluate

# 替换: 原 MADDPGWithAttention → 新 MADDPGNoiseWithAttention (含信号分支)
from model.maddpg_noise_att_lstm import MADDPGNoiseWithAttention
from env.sim_env_rvo import UAVEnv
from buffer.buffer import MultiAgentReplayBuffer
from buffer.PERbuffer import PERMultiAgentReplayBuffer
import time
import pandas as pd
import os
import sys
import matplotlib
matplotlib.use('Agg')  # 无显示环境
import matplotlib.pyplot as plt
import warnings
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import subprocess
from UAVTask import UAVTaskEvaluator


warnings.filterwarnings('ignore')


# ========== 1. 日志目录改为绝对路径 + 必存在校验 (修复 FileNotFoundError 崩溃) ==========
def create_unique_log_dir(base_dir="runs_noise/UAV_training"):
    base_dir_abs = os.path.abspath(base_dir)
    os.makedirs(base_dir_abs, exist_ok=True)
    idx = 1
    while True:
        log_dir = f"{base_dir_abs}{idx:02d}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            # 立刻验证目录存在
            assert os.path.isdir(log_dir), f"log_dir create failed: {log_dir}"
            return log_dir
        idx += 1


def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def obs_list_to_state_vector(obs):
    state = np.hstack([np.ravel(o) for o in obs])
    return state


def save_image(env_render, filename):
    image = Image.fromarray(env_render, 'RGBA')
    image = image.convert('RGB')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    image.save(filename)


# ========== 2. SummaryWriter.add_scalar 容错封装 (修复异步写线程崩溃) ==========
def safe_add_scalar(writer, tag, scalar_value, global_step=None):
    try:
        writer.add_scalar(tag, scalar_value, global_step)
        writer.flush()
    except Exception as e:
        print(f"[WARN] tensorboard write {tag} failed: {type(e).__name__}: {e}", file=sys.stderr)


def main(n_games=5000, batch_episodes=100, max_steps=110,
         alpha=0.0001, beta=0.001,
         evaluate=False,
         chkpt_dir='tmp_noise/maddpgwithatt/',
         signal_hidden=32,          # SignalEncoder 隐层维度 (CPU 友好,原 64→32)
         learn_every=20,            # 每 N 步做一次 learn (缓解 CPU 瓶颈)
         per_batch=128,             # PER 采样 batch_size (原 256→128)
         n_signal_samples=16,       # 与 env/sim_env_rvo.py 保持一致
         obs_agt=42,                # 围捕者 obs 维度: 26 + 16
         obs_tar=23,                # 逃跑者 obs 维度: 不变
         n_hunters=3,
         log_file='training_log_noise.txt',
         reward_curve_file='reward_curves_noise.png'):
    # ---- 日志文件 ----
    log_file_abs = os.path.abspath(log_file)
    log_f = open(log_file_abs, 'a', buffering=1)
    def tee(msg):
        print(msg, flush=True)
        log_f.write(msg + '\n')

    # ---- 环境初始化 ----
    env = UAVEnv()
    log_dir = create_unique_log_dir()
    tee(f"[INFO] tensorboard log_dir = {log_dir}")
    writer = SummaryWriter(log_dir=log_dir, flush_secs=10)

    n_agents = env.num_agents
    actor_dims = []
    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)
    tee(f"[INFO] actor_dims={actor_dims}, critic_dims={critic_dims} "
        f"(hunter obs = {obs_agt}, target obs = {obs_tar})")
    n_actions = 2
    lstm_hidden_dim = 256

    # ---- 新: 用带信号分支的 MADDPGNoiseWithAttention ----
    chkpt_dir_abs = os.path.abspath(chkpt_dir)
    os.makedirs(chkpt_dir_abs, exist_ok=True)
    maddpg_agents = MADDPGNoiseWithAttention(
        actor_dims, critic_dims, n_agents, n_actions,
        alpha=alpha, beta=beta, scenario='UAV_Round_up',
        chkpt_dir=chkpt_dir_abs + '/',
        obs_agt=obs_agt, obs_tar=obs_tar,
        n_signal_samples=n_signal_samples,
        signal_hidden=signal_hidden,
        n_hunters=n_hunters,
    )

    # ---- 超参记录 ----
    hyperparams_path = os.path.join(chkpt_dir_abs, 'hyperparameters.txt')
    with open(hyperparams_path, 'w') as f:
        f.write(f'alpha: {alpha}\n')
        f.write(f'beta: {beta}\n')
        f.write(f'n_signal_samples: {n_signal_samples}\n')
        f.write(f'signal_hidden: {signal_hidden}\n')
        f.write(f'learn_every: {learn_every}\n')
        f.write(f'per_batch: {per_batch}\n')
        f.write(f'obs_agt: {obs_agt}\n')
        f.write(f'obs_tar: {obs_tar}\n')
    tee(f"[INFO] checkpoint dir = {chkpt_dir_abs}  (含 scenario=UAV_Round_up 子目录)")

    # ---- PER buffer (batch_size 可调) ----
    memory = PERMultiAgentReplayBuffer(1000000, critic_dims, actor_dims,
                        n_actions, n_agents, batch_size=per_batch)

    BATCH_SIZE = batch_episodes
    N_GAMES = n_games
    MAX_STEPS = max_steps
    total_steps = 0
    score_history = []
    target_score_history = []
    best_score = -1000
    window_size = 100
    success_evaluator_each = []

    # ---- 尝试加载已有 checkpoint (静默失败, 从头训练也 OK) ----
    try:
        maddpg_agents.load_checkpoint()
        tee("[INFO] pre-trained checkpoint loaded (if any)")
    except Exception as e:
        tee(f"[INFO] no pre-trained checkpoint loaded: {type(e).__name__}: {e}")

    if evaluate:
        tee('----  evaluating  ----')
    else:
        tee('----training start----')

    evaluator = UAVTaskEvaluator([0, 0, 0], max_time_steps=MAX_STEPS,
                                 dones=[False] * n_agents)

    # ========================== 主训练循环 ==========================
    i = 0
    while i < N_GAMES:
        with tqdm(total=BATCH_SIZE,
                  desc=f"Batch (Total {i}/{N_GAMES})",
                  unit="episode") as pbar:
            for _ in range(BATCH_SIZE):
                if i >= N_GAMES:
                    break

                obs = env.reset()
                score = 0
                score_target = 0
                dones = [False] * n_agents
                episode_step = 0

                while not any(dones):
                    if evaluate:
                        env_render = env.render()
                        if episode_step % 10 == 0:
                            filename = f'images/episode_{i}_step_{episode_step}.png'
                            os.makedirs(os.path.dirname(filename), exist_ok=True)
                            save_image(env_render, filename)

                    actions = maddpg_agents.choose_action(obs, total_steps, evaluate)
                    obs_, rewards, dones, collision_info, mul_pos = env.step(actions)

                    success_evaluator_total, total_counts = evaluator.success_evaluate(dones, i)

                    state = obs_list_to_state_vector(obs)
                    state_ = obs_list_to_state_vector(obs_)

                    if episode_step >= MAX_STEPS:
                        dones = [True] * n_agents

                    memory.store_transition(obs, state, actions, rewards, obs_, state_, dones)

                    # ---- 新: learn_every 可控 (原固定 10) ----
                    if total_steps % learn_every == 0 and not evaluate:
                        maddpg_agents.learn(memory, total_steps, state)

                    obs = obs_
                    score += sum(rewards[0:3])
                    score_target += rewards[-1]
                    total_steps += 1
                    episode_step += 1

                # 成功率报告
                if success_evaluator_total and episode_step < MAX_STEPS:
                    msg = f"第 {i} 回合, 总成功率: {success_evaluator_total * 100:.2f}%"
                    tqdm.write(msg)
                    tee(msg)

                score_history.append(score)
                target_score_history.append(score_target)

                # ---- TensorBoard 容错写入 ----
                safe_add_scalar(writer, "UAV Reward", score, i)
                safe_add_scalar(writer, "Target Reward", score_target, i)

                pbar.update(1)
                pbar.set_postfix({
                    "ep_R": f"{score:.2f}",
                    "tgt_R": f"{score_target:.2f}"
                })
                i += 1

            # ========== 每个 batch 汇总 ==========
            avg_score = np.mean(score_history[-BATCH_SIZE:])
            avg_target_score = np.mean(target_score_history[-BATCH_SIZE:])

            pbar.close()
            batch_idx = i // BATCH_SIZE
            summary = (f'Batch {batch_idx} done. '
                       f'avg_score={avg_score:.2f}, '
                       f'avg_target={avg_target_score:.2f}, '
                       f'total_steps={total_steps}')
            tqdm.write(summary)
            tee(summary)

            # ========== 保存 checkpoint ==========
            # 策略: 每个 batch 都保存一次,但 best 更新时额外打印并触发 evaluate
            save_ok = False
            try:
                maddpg_agents.save_checkpoint()
                # 校验: 列出 scenario 子目录下是否真有文件
                actual_chkpt_sub = os.path.join(chkpt_dir_abs, 'UAV_Round_up')
                if os.path.isdir(actual_chkpt_sub):
                    files = os.listdir(actual_chkpt_sub)
                    pt_files = [f for f in files if not f.endswith('.txt')]
                    if pt_files:
                        save_ok = True
                        tee(f'[INFO] checkpoint verified @ {actual_chkpt_sub}: '
                            f'{len(pt_files)} files, e.g. {pt_files[0]}')
                    else:
                        tee(f'[WARN] checkpoint dir empty? {actual_chkpt_sub} -> {files}')
                else:
                    tee(f'[WARN] checkpoint subdir missing: {actual_chkpt_sub}')
            except Exception as e:
                tee(f'[ERROR] save_checkpoint FAILED: {type(e).__name__}: {e}')

            if avg_score > best_score:
                tee(f'New best avg score {avg_score:.2f} > previous best {best_score:.2f}. '
                    f'(saved={save_ok})')
                best_score = avg_score
                # 尝试启动 evaluate,失败不影响训练
                try:
                    process = subprocess.Popen(
                        ['python', 'evaluate_arvp.py'],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    time.sleep(10)
                    process.terminate()
                    process.wait(timeout=5)
                except Exception as e:
                    tee(f'[WARN] evaluate subprocess skipped: {type(e).__name__}: {e}')

    # ========================== 收尾: 奖励曲线 + 关闭 ==========================
    tee('[INFO] training finished, drawing reward curves...')
    if len(score_history) >= window_size:
        score_history_ma = moving_average(score_history, window_size)
        target_score_history_ma = moving_average(target_score_history, window_size)

    plt.figure(figsize=(12, 6))
    plt.plot(score_history, label='Raw Total Reward', alpha=0.3)
    if len(score_history) >= window_size:
        plt.plot(range(window_size - 1, len(score_history)),
                 score_history_ma, label='Smoothed Total Reward')
    plt.title('Total Reward per Episode (ARVP + Sonar Signal + GRU+Attention)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    curve_abs = os.path.abspath(reward_curve_file)
    plt.savefig(curve_abs, dpi=120)
    plt.close()
    tee(f'[INFO] reward curve saved to {curve_abs}')

    try:
        writer.close()
    except Exception as e:
        print(f"[WARN] writer.close failed: {e}", file=sys.stderr)

    tee(f'[INFO] training log saved to {log_file_abs}')
    log_f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ARVP + Sonar Signal Training")
    parser.add_argument('--n_games', type=int, default=5000)
    parser.add_argument('--batch', type=int, default=100, help='episodes per batch (report interval)')
    parser.add_argument('--max_steps', type=int, default=110)
    parser.add_argument('--alpha', type=float, default=0.0001)
    parser.add_argument('--beta', type=float, default=0.001)
    parser.add_argument('--signal_hidden', type=int, default=32)
    parser.add_argument('--learn_every', type=int, default=20)
    parser.add_argument('--per_batch', type=int, default=128)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--chkpt_dir', type=str, default='tmp_noise/maddpgwithatt/')
    parser.add_argument('--log_file', type=str, default='training_log_noise.txt')
    parser.add_argument('--reward_curve', type=str, default='reward_curves_noise.png')
    args = parser.parse_args()

    main(
        n_games=args.n_games,
        batch_episodes=args.batch,
        max_steps=args.max_steps,
        alpha=args.alpha,
        beta=args.beta,
        evaluate=args.evaluate,
        chkpt_dir=args.chkpt_dir,
        signal_hidden=args.signal_hidden,
        learn_every=args.learn_every,
        per_batch=args.per_batch,
        log_file=args.log_file,
        reward_curve_file=args.reward_curve,
    )
