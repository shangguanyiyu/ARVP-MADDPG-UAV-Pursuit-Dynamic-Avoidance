import numpy as np
from sympy import evaluate

from model.maddpg_att import MADDPGWithAttention
from env.sim_env_rvo import UAVEnv
from buffer.buffer import MultiAgentReplayBuffer
from buffer.PERbuffer import PERMultiAgentReplayBuffer
import time
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import subprocess
from UAVTask import UAVTaskEvaluator


# from kf import KalmanFilter
warnings.filterwarnings('ignore')


def create_unique_log_dir(base_dir="runs/UAV_training"):
    idx = 1
    while True:
        log_dir = f"{base_dir}{idx:02d}"  # 例如 UAV_training01, UAV_training02
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            return log_dir
        idx += 1

def moving_average(data, window_size=100):
    """计算滑动窗口均值"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def obs_list_to_state_vector(obs):
    state = np.hstack([np.ravel(o) for o in obs])
    return state

def save_image(env_render, filename):
    # Convert the RGBA buffer to an RGB image
    image = Image.fromarray(env_render, 'RGBA')  # Use 'RGBA' mode since the buffer includes transparency
    image = image.convert('RGB')  # Convert to 'RGB' if you don't need transparency

    image.save(filename)

if __name__ == '__main__':

    env = UAVEnv()
    log_dir = create_unique_log_dir()
    writer = SummaryWriter(log_dir=log_dir)
    # =========================================================================
    # APF 方案一：CSV 训练日志目录和文件创建
    #   保存字段: episode, total_reward(hunters), target_reward, success(0/1),
    #             episode_steps, batch_avg_score, batch_avg_target_score
    # =========================================================================
    csv_log_dir = os.path.join(log_dir, "csv_logs")
    os.makedirs(csv_log_dir, exist_ok=True)
    csv_train_file = os.path.join(csv_log_dir, "train_log_apf_5000.csv")
    csv_header = ["episode", "total_hunter_reward", "target_reward",
                  "success_flag", "episode_steps", "batch_avg_score",
                  "batch_avg_target_score"]
    with open(csv_train_file, "w", newline="") as f_csv:
        pd.DataFrame(columns=csv_header).to_csv(f_csv, index=False)
    csv_rows = []

    n_agents = env.num_agents
    actor_dims = []
    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)
    # 3D + APF observation dims: hunter=50, target=44 (derived from env.observation_space)
    obs_agt = actor_dims[0]
    obs_tar = actor_dims[-1]

    n_actions = 3  # 3D actions (ax, ay, az)
    lstm_hidden_dim =256
    alpha = 0.00001 #0.00001
    beta = 0.001 # 0.001
    chkpt_dir = 'tmp_avoid_dynamic/maddpgwithatt/'
    maddpg_agents = MADDPGWithAttention(actor_dims, critic_dims, n_agents, n_actions,
                           obs_agt=obs_agt, obs_tar=obs_tar,
                           alpha=alpha, beta=beta, scenario='UAV_Round_up', chkpt_dir=chkpt_dir) # alpha=0.00001, beta=0.00005

    # maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions,
    #                        fc1=128, fc2=128,
    #                        alpha=alpha, beta=beta, scenario='UAV_Round_up',
    #                        chkpt_dir=chkpt_dir)
    # 保存超参数到上一级目录
    hyperparams_path = os.path.join(chkpt_dir, 'hyperparameters.txt')

    with open(hyperparams_path, 'w') as f:
        f.write(f'alpha: {alpha}\n')
        f.write(f'beta: {beta}\n')
    #
    # memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims,
    #                     n_actions, n_agents, batch_size=256)
    memory = PERMultiAgentReplayBuffer(1000000, critic_dims, actor_dims,
                        n_actions, n_agents, batch_size=256)

    BATCH_SIZE = 100  # 每个 batch 包含的回合数
    N_GAMES = 5000
    MAX_STEPS = 110
    total_steps = 0
    score_history = []
    target_score_history = []
    evaluate = False
    best_score = -1000
    window_size = 100
    success_evaluator_each = []
    print(chkpt_dir)
    try:
        maddpg_agents.load_checkpoint()
    except Exception as e:
        print(f'[warn] load_checkpoint skipped (no/incompatible checkpoint): {e}')
    if evaluate:
        try:
            maddpg_agents.load_checkpoint()
        except Exception as e:
            print(f'[warn] load_checkpoint skipped (no/incompatible checkpoint): {e}')
        print('----  evaluating  ----')
    else:
        print('----training start----')
    evaluator = UAVTaskEvaluator([0, 0, 0], max_time_steps=MAX_STEPS, dones=[False, False, False, False]) # 初始化评测类

    # 主训练循环
    i = 0
    while i < N_GAMES:
        # if i >= 100:
        #     break
        with tqdm(total=BATCH_SIZE, desc=f"Batch Progress (Total {i}/{N_GAMES})", unit="episode") as pbar:
            for _ in range(BATCH_SIZE):
                if i >= N_GAMES:
                    break  # 确保不会超过总回合数

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
                    # print(actions)
                    obs_, rewards, dones, collision_info, mul_pos = env.step(actions)
                    # for agt_id, done in enumerate(dones):
                    #     success_rate, agent_success_count, agent_total_rounds = evaluator.success_evaluate(done)

                        # # 打印每个无人机的成功率（每 100 回合）
                        # if success_rate is not None:
                        #     print(
                        #         f"第 {agent_total_rounds} 回合，无人机 {agt_id} 成功率: {success_rate * 100:.2f}%")
                    success_evaluator_total, total_counts = evaluator.success_evaluate(dones, i)


                    state = obs_list_to_state_vector(obs)
                    state_ = obs_list_to_state_vector(obs_)

                    if episode_step >= MAX_STEPS:
                        dones = [True] * n_agents

                    memory.store_transition(obs, state, actions, rewards, obs_, state_, dones)

                    if total_steps % 10 == 0 and not evaluate:
                        maddpg_agents.learn(memory, total_steps, state)
                        # maddpg_agents.learn(memory, total_steps)

                    obs = obs_
                    score += sum(rewards[0:3])
                    score_target += rewards[-1]
                    total_steps += 1
                    episode_step += 1
                # print(success_evaluator_total)
                if success_evaluator_total and episode_step < MAX_STEPS:
                    # success_evaluator_each.append(success_evaluator_total)
                    print(f"第 {i} 回合，总的成功率: {success_evaluator_total * 100:.2f}%")
                # 记录奖励值
                score_history.append(score)
                target_score_history.append(score_target)
                # 使用 TensorBoard 记录奖励
                writer.add_scalar("UAV Reward", score, i)
                writer.add_scalar("Target Reward", score_target, i)

                # =================================================================
                # APF 方案一：CSV 日志记录（每回合）
                # =================================================================
                success_flag = 1 if (success_evaluator_total and episode_step < MAX_STEPS) else 0
                csv_rows.append({
                    "episode": i,
                    "total_hunter_reward": float(score),
                    "target_reward": float(score_target),
                    "success_flag": success_flag,
                    "episode_steps": int(episode_step),
                    "batch_avg_score": None,   # filled at batch end
                    "batch_avg_target_score": None,
                })

                # 更新进度条
                pbar.update(1)
                # 间隔一定回合运行一次评测代码


                pbar.set_postfix({
                    "Episode Reward": f"{score:.2f}",
                    "Target Reward": f"{score_target:.2f}"
                })
                i += 1


            # # 每隔一个 batch 计算 avg_score
            # avg_score = np.mean(score_history[-BATCH_SIZE:])
            # avg_target_score = np.mean(target_score_history[-BATCH_SIZE:])
            #
            # # 如果 avg_score 比 best_score 大，保存模型
            # if avg_score > best_score:
            #     pbar.close()  # 关闭当前进度条
            #     tqdm.write(f' best avg score {avg_score:.2f} > {best_score:.2f}, saving models...')
            #     maddpg_agents.save_checkpoint()
            #     best_score = avg_score
            #     # pbar = tqdm(total=BATCH_SIZE, desc=f"Batch Progress (Total {i}/{N_GAMES})", unit="episode", leave=True)

            # 每隔一个 batch 计算 avg_score
            avg_score = np.mean(score_history[-BATCH_SIZE:])
            avg_target_score = np.mean(target_score_history[-BATCH_SIZE:])

            # =================================================================
            # APF 方案一：回填本 batch 内最后一回合的 avg_score 作为标记, 并 flush CSV
            # =================================================================
            if csv_rows:
                for row in csv_rows[-BATCH_SIZE:]:
                    row["batch_avg_score"] = float(avg_score)
                    row["batch_avg_target_score"] = float(avg_target_score)
                # flush accumulated rows to disk every batch
                with open(csv_train_file, "a", newline="") as f_csv:
                    pd.DataFrame(csv_rows[-BATCH_SIZE:])[csv_header].to_csv(
                        f_csv, header=False, index=False)
                tqdm.write(f'CSV training log flushed to {csv_train_file} (total rows so far: {len(csv_rows)})')

            # 打印当前批次的进度和平均奖励
            pbar.close()  # 关闭当前进度条
            tqdm.write(f'Batch {i // BATCH_SIZE} completed.')
            tqdm.write(f'Average Score: {avg_score:.2f}, Average Target Score: {avg_target_score:.2f}')

            # 保存模型（每个 batch 都保存一次）
            
            tqdm.write(f'Model saved at batch {i // BATCH_SIZE}.')

            # 如果 avg_score 比 best_score 大，更新 best_score
            if avg_score > best_score:

                tqdm.write(f'New best avg score {avg_score:.2f} > previous best score {best_score:.2f}.')
                maddpg_agents.save_checkpoint()
                best_score = avg_score
                # 注释掉 evaluate_arvp.py 子进程调用，避免训练中频繁启动浪费时间
                # process = subprocess.Popen(['python', 'evaluate_arvp.py'])
                # time.sleep(10)
                # process.terminate()
                # process.wait()

    # 计算滑动均值
    if len(score_history) >= window_size:
        score_history_ma = moving_average(score_history, window_size)
        target_score_history_ma = moving_average(target_score_history, window_size)

    # 绘制奖励曲线
    plt.figure(figsize=(12, 6))

    # 总奖励曲线
    plt.plot(score_history, label='Raw Total Reward', alpha=0.3)  # 原始奖励曲线
    if len(score_history) >= window_size:
        plt.plot(range(window_size - 1, len(score_history)), score_history_ma, label='Smoothed Total Reward')  # 滑动均值曲线
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid()

    # 保存图像
    plt.tight_layout()
    plt.savefig('reward_curves.png')
    # =========================================================================
    # APF 方案一：同时保存一份奖励曲线图到 CSV 日志目录确保可提交
    # =========================================================================
    reward_curves_in_logdir = os.path.join(csv_log_dir, "reward_curves_5000epi.png")
    plt.savefig(reward_curves_in_logdir)
    tqdm.write(f'reward_curves.png saved to ./reward_curves.png and {reward_curves_in_logdir}')
    try:
        plt.show()
    except Exception:
        pass  # headless environment: safe to ignore

    writer.close()
