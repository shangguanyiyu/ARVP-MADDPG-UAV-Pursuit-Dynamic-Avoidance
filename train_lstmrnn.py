import numpy as np
from sympy import evaluate

from model.maddpg_lstm_pre import MADDPGWithAttentionLSTMPRE
from env.sim_env_rvo import UAVEnv
from buffer.buffer import MultiAgentReplayBuffer
from buffer.RNNbuffer import RNNMultiAgentReplayBuffer
from buffer.RNNPREbuffer import RNNPERMultiAgentReplayBuffer
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
from model.networks_att_lstm import CriticNetwork_1 as Critic
from model.networks_att_lstm import ActorNetwork as Actor
import torch


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

    n_agents = env.num_agents
    actor_dims = []
    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)

    n_actions = 2
    lstm_hidden_dim =256
    lstm_num_layers = 4
    maddpg_agents = MADDPGWithAttentionLSTMPRE(actor_dims, critic_dims, n_agents, n_actions,
                           fc1=128, fc2=128, lstm_hidden_dim=lstm_hidden_dim,
                           alpha=0.01, beta=0.01, scenario='UAV_Round_up', chkpt_dir='tmp_test/maddpgwithatt/', num_layers=lstm_num_layers) # alpha=0.00001, beta=0.00005

    # memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims,
    #                     n_actions, n_agents, batch_size=256)
    # memory = RNNMultiAgentReplayBuffer(100000, critic_dims, actor_dims,
    #                     n_actions, n_agents, batch_size=256, hidden_state_dim=lstm_hidden_dim)
    memory = RNNPERMultiAgentReplayBuffer(100000, critic_dims, actor_dims,
                        n_actions, n_agents, batch_size=256, hidden_state_dim=lstm_hidden_dim, lstm_num_layers=lstm_num_layers)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # hidden_state_critic = memory.init_hidden_memory(Critic.hidden_dim, Critic.lstm_hidden_dim, device)
    hidden_state_actor = memory.init_hidden_memory(lstm_num_layers, 1,lstm_hidden_dim, device)

    BATCH_SIZE = 100  # 每个 batch 包含的回合数
    N_GAMES = 50000
    MAX_STEPS = 110
    total_steps = 0
    score_history = []
    target_score_history = []
    evaluate = False
    best_score = -10000
    window_size = 100
    success_evaluator_each = []
    # maddpg_agents.load_checkpoint()
    if evaluate:
        maddpg_agents.load_checkpoint()
        print('----  evaluating  ----')
    else:
        print('----training start----')
    evaluator = UAVTaskEvaluator([0, 0, 0], max_time_steps=MAX_STEPS, dones=[False, False, False, False]) # 初始化评测类

    # 主训练循环
    i = 0
    while i < N_GAMES:
        with tqdm(total=BATCH_SIZE, desc=f"Batch Progress (Total {i}/{N_GAMES})", unit="episode") as pbar:
            for _ in range(BATCH_SIZE):
                if i >= N_GAMES:
                    break  # 确保不会超过总回合数

                obs = env.reset()
                hidden_state_actor = memory.init_hidden_memory(lstm_num_layers, 1, lstm_hidden_dim, device)
                hidden_state_actor_list = [hidden_state_actor for _ in range(4)]
                # hidden_state_eatch_actor = memory.init_hidden_memory(1, 1, lstm_hidden_dim, device)
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

                    actions, hidden_state_actor_list, next_hidden_state_actor_list= maddpg_agents.choose_action(obs, total_steps, evaluate, hidden_state_actor_list)

                    # print(next_hidden_state_actor[0].shape,next_hidden_state_actor[1].shape) [1,1,256]
                    obs_, rewards, dones, collision_info, mul_pos = env.step(actions) # step2 基于动力学模型
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

                    memory.store_transition(obs, state, actions, rewards, obs_, state_, dones, hidden_state_actor_list, next_hidden_state_actor_list)

                    if total_steps % 10 == 0 and not evaluate:
                        maddpg_agents.learn(memory, total_steps, state, hidden_state_actor_list, total_steps)

                    obs = obs_
                    hidden_state_actor_list = next_hidden_state_actor_list
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
                process = subprocess.Popen(['python', 'evaluate_rnn.py'])

                # 模拟主程序运行
                time.sleep(10)  # 假设主程序运行 10 秒

                # 终止子进程
                process.terminate()  # 发送 SIGTERM 信号
                # process.kill()  # 如果需要强制终止，可以使用 SIGKILL

                # 等待子进程结束
                process.wait()

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
    plt.show()

    writer.close()
