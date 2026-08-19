"""
train_noise_arvp.py (trae/arvpNoise 新增)
================================================
ARVP + 声呐信号感知 (FFT + GRU + 多头自注意力) 训练入口
- 环境: env/sim_env_rvo.py (obs 维度已扩到 42, 末 16 维为时域声呐采样)
- 模型: model/maddpg_noise_att_lstm.py
- 经验回放: PER (优先经验回放)
- 训练目标: 5000 episodes, 让网络对围捕目标/动态障碍的信号特征越来越敏感
- 输出: runs_noise/UAV_trainingXX (TB 日志), reward_curves_noise.png, training_log_noise.txt
- reward 机制保持不变, 仅 obs / NN 结构扩展
"""
import numpy as np
import sys
import time
import os
import io
import matplotlib.pyplot as plt
import warnings
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 确保工作目录可被 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.sim_env_rvo import UAVEnv
from model.maddpg_noise_att_lstm import MADDPGNoiseWithAttention
from buffer.PERbuffer import PERMultiAgentReplayBuffer
from UAVTask import UAVTaskEvaluator

warnings.filterwarnings('ignore')

# 同时把 stdout/stderr 写到日志文件, 方便回看
LOG_FILE = 'training_log_noise.txt'
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


def create_unique_log_dir(base_dir="runs_noise/UAV_training"):
    idx = 1
    while True:
        log_dir = f"{base_dir}{idx:02d}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            return log_dir
        idx += 1


def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def obs_list_to_state_vector(obs):
    return np.hstack([np.ravel(o) for o in obs])


def save_image(env_render, filename):
    image = Image.fromarray(env_render, 'RGBA').convert('RGB')
    image.save(filename)


def main(n_games=5000, batch_episodes=100, max_steps=110,
         alpha=0.0001, beta=0.001, evaluate=False,
         chkpt_dir='tmp_noise/maddpgwithatt/'):
    log_fp = open(LOG_FILE, 'a', buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_fp)
    sys.stderr = Tee(sys.__stderr__, log_fp)

    print(f"\n========== 训练启动 {time.strftime('%Y-%m-%d %H:%M:%S')} ==========")
    print(f"N_GAMES={n_games}, MAX_STEPS={max_steps}, BATCH={batch_episodes}")
    print(f"alpha={alpha}, beta={beta}, evaluate={evaluate}")
    print(f"chkpt_dir={chkpt_dir}")

    env = UAVEnv()
    print(f"observation_space: {[(k, v.shape) for k, v in env.observation_space.items()]}")
    print(f"n_signal_samples={env.n_signal_samples}, target_freqs={env.target_freqs}, obstacle_freqs={env.obstacle_freqs}")

    log_dir = create_unique_log_dir()
    writer = SummaryWriter(log_dir=log_dir)
    print(f"tensorboard log_dir = {log_dir}")

    n_agents = env.num_agents
    actor_dims = [env.observation_space[ag].shape[0] for ag in env.observation_space]
    critic_dims = sum(actor_dims)
    n_actions = 2
    obs_agt = actor_dims[0]            # 42
    obs_tar = actor_dims[-1]           # 23
    n_signal_samples = env.n_signal_samples  # 16

    maddpg_agents = MADDPGNoiseWithAttention(
        actor_dims, critic_dims, n_agents, n_actions,
        alpha=alpha, beta=beta, scenario='UAV_Round_up',
        chkpt_dir=chkpt_dir, obs_agt=obs_agt, obs_tar=obs_tar,
        n_signal_samples=n_signal_samples, signal_hidden=64, n_hunters=3,
    )

    memory = PERMultiAgentReplayBuffer(1000000, critic_dims, actor_dims,
                                       n_actions, n_agents, batch_size=256)

    total_steps = 0
    score_history = []
    target_score_history = []
    best_score = -1e9
    window_size = 100

    # 保存超参数
    os.makedirs(chkpt_dir, exist_ok=True)
    with open(os.path.join(chkpt_dir, 'hyperparameters.txt'), 'w') as f:
        f.write(f'alpha: {alpha}\n')
        f.write(f'beta: {beta}\n')
        f.write(f'n_games: {n_games}\n')
        f.write(f'max_steps: {max_steps}\n')
        f.write(f'n_signal_samples: {n_signal_samples}\n')
        f.write(f'obs_agt: {obs_agt}\n')
        f.write(f'obs_tar: {obs_tar}\n')
        f.write(f'target_freqs: {env.target_freqs.tolist()}\n')
        f.write(f'obstacle_freqs: {env.obstacle_freqs.tolist()}\n')

    if evaluate:
        maddpg_agents.load_checkpoint()
        print('----  evaluating  ----')
    else:
        print('----training start----')

    evaluator = UAVTaskEvaluator([0, 0, 0], max_time_steps=max_steps, dones=[False]*4)

    i = 0
    while i < n_games:
        with tqdm(total=batch_episodes, desc=f"Batch (Total {i}/{n_games})", unit="ep") as pbar:
            for _ in range(batch_episodes):
                if i >= n_games:
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
                            filename = f'images_noise/episode_{i}_step_{episode_step}.png'
                            os.makedirs(os.path.dirname(filename), exist_ok=True)
                            save_image(env_render, filename)
                    actions = maddpg_agents.choose_action(obs, total_steps, evaluate)
                    obs_, rewards, dones, collision_info, mul_pos = env.step(actions)
                    success_evaluator_total, _ = evaluator.success_evaluate(dones, i)
                    state = obs_list_to_state_vector(obs)
                    state_ = obs_list_to_state_vector(obs_)
                    if episode_step >= max_steps:
                        dones = [True] * n_agents
                    memory.store_transition(obs, state, actions, rewards, obs_, state_, dones)
                    if total_steps % 10 == 0 and not evaluate:
                        maddpg_agents.learn(memory, total_steps, state)
                    obs = obs_
                    score += sum(rewards[0:3])
                    score_target += rewards[-1]
                    total_steps += 1
                    episode_step += 1
                if success_evaluator_total and episode_step < max_steps:
                    print(f"第 {i} 回合, 总成功率: {success_evaluator_total*100:.2f}%")
                score_history.append(score)
                target_score_history.append(score_target)
                writer.add_scalar("UAV Reward", score, i)
                writer.add_scalar("Target Reward", score_target, i)
                pbar.update(1)
                pbar.set_postfix({"ep_R": f"{score:.2f}", "tgt_R": f"{score_target:.2f}"})
                i += 1

            avg_score = np.mean(score_history[-batch_episodes:])
            avg_target_score = np.mean(target_score_history[-batch_episodes:])
            pbar.close()
            print(f"Batch {i // batch_episodes} done. avg_score={avg_score:.2f}, avg_target={avg_target_score:.2f}, total_steps={total_steps}")
            if avg_score > best_score:
                print(f"new best avg {avg_score:.2f} > prev {best_score:.2f}, saving...")
                maddpg_agents.save_checkpoint()
                best_score = avg_score

            # 实时刷新 reward 曲线
            try:
                _plot_reward_curves(score_history, target_score_history, window_size)
            except Exception as e:
                print(f"plot err: {e}")

    # 最终曲线
    _plot_reward_curves(score_history, target_score_history, window_size)
    writer.close()
    print(f"========== 训练结束 {time.strftime('%Y-%m-%d %H:%M:%S')} ==========")
    print(f"best_score={best_score:.2f}, total episodes={len(score_history)}, total_steps={total_steps}")
    log_fp.close()


def _plot_reward_curves(score_history, target_score_history, window_size):
    plt.figure(figsize=(12, 6))
    plt.plot(score_history, label='Raw Total Reward', alpha=0.3)
    if len(score_history) >= window_size:
        ma = moving_average(score_history, window_size)
        plt.plot(range(window_size - 1, len(score_history)), ma, label=f'Smoothed (w={window_size})')
    plt.title('ARVP+Noise Signal Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (3 hunters)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('reward_curves_noise.png')
    plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_games', type=int, default=5000)
    parser.add_argument('--max_steps', type=int, default=110)
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=0.0001)
    parser.add_argument('--beta', type=float, default=0.001)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--chkpt_dir', type=str, default='tmp_noise/maddpgwithatt/')
    args = parser.parse_args()
    main(n_games=args.n_games, batch_episodes=args.batch, max_steps=args.max_steps,
         alpha=args.alpha, beta=args.beta, evaluate=args.evaluate,
         chkpt_dir=args.chkpt_dir)
