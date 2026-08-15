"""
origin3D 训练脚本：原生 3D MADDPG-LSTM-Att-PER 基线 (无方案特定增强)
对应环境 env/sim_env_rvo.py (真3D环境)
关键改动：
  - 观测维度：agent 56 (3D), target 50 (3D)  — 取自 env.observation_space
  - 动作维度：3 (ax, ay, az)
  - N_GAMES=5000
  - 自动创建 log_arvp_origin3D/ 目录保存奖励曲线 PNG + score_history.csv + hyperparameters.csv
  - 手动输入 stage 参数，目录使用 "当前日期+时间_stage参数_" 前缀
  - 模型 / 日志 / runs 全部放到带日期时间前缀的子目录中
  - 参数通过 JSON 中心化保存：_last_arvp_origin3D_run.json
"""
import numpy as np
from sympy import evaluate

from model.maddpg_att import MADDPGWithAttention
from env.sim_env_rvo import UAVEnv
from buffer.buffer import MultiAgentReplayBuffer
from buffer.PERbuffer import PERMultiAgentReplayBuffer
import time
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # 无显示环境也能保存图
import matplotlib.pyplot as plt
import warnings
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import subprocess
import json
import sys
from UAVTask import UAVTaskEvaluator
from datetime import datetime


RUN_CONFIG_FILENAME = "_last_arvp_origin3D_run.json"  # 参数传递文件（根目录下）


def save_run_config(config: dict) -> str:
    """把本次训练的关键路径/参数写到 JSON，供 evaluate 端直接读取（参数中心化）"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), RUN_CONFIG_FILENAME)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return path


warnings.filterwarnings('ignore')


def get_run_prefix(stage: str) -> str:
    """生成当前日期+时间_stage参数的前缀，例如 0814_153020_stage1"""
    now = datetime.now()
    ts = now.strftime("%m%d_%H%M%S")
    return f"{ts}_{stage}"


def create_unique_log_dir(base_dir: str) -> str:
    """在 base_dir 下创建 UAV_training01/02/... 的唯一子目录"""
    idx = 1
    while True:
        log_dir = f"{base_dir}{idx:02d}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
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


if __name__ == '__main__':
    # ================== 手动输入 stage 参数 ==================
    DEFAULT_STAGE = "stage1"
    STAGE = DEFAULT_STAGE
    print(f"[origin3D-基线] 使用 stage = {STAGE}")

    # 生成日期时间+stage 前缀（全局唯一标识）
    RUN_PREFIX = get_run_prefix(STAGE)
    print(f"[origin3D-基线] RUN_PREFIX = {RUN_PREFIX}")

    # ================== runs/下创建带前缀的目录，UAV_trainingXX放其中 ==================
    RUNS_PARENT_DIR = os.path.join("runs", f"{RUN_PREFIX}_arvp_origin3D")
    os.makedirs(RUNS_PARENT_DIR, exist_ok=True)
    RUNS_UAV_BASE = os.path.join(RUNS_PARENT_DIR, "UAV_training")

    # ================== log_arvp_origin3D/下创建带前缀的目录 ==================
    LOG_DIR = os.path.join("log_arvp_origin3D", f"{RUN_PREFIX}_arvp_origin3D")
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"[origin3D-基线] 日志目录: {LOG_DIR}/")

    # ================== 模型保存在 tmp_avoid_dynamic/maddpgwithatt_origin3D/日期时间_stage_UAV_Round_up ==================
    BASE_CHKPT_DIR = os.path.join("tmp_avoid_dynamic", "maddpgwithatt_origin3D")
    os.makedirs(BASE_CHKPT_DIR, exist_ok=True)
    CHKPT_DIR = BASE_CHKPT_DIR + os.sep  # 结尾加分隔符，和旧代码保持一致
    SCENARIO_NAME = f"{RUN_PREFIX}_UAV_Round_up"
    MODEL_SAVE_DIR = os.path.join(BASE_CHKPT_DIR, SCENARIO_NAME)  # 用于日志打印

    # ================== 参数中心化：保存 JSON 配置供 evaluate 端直接读取 ==================
    RUN_CONFIG = {
        # --- 路径相关 ---
        "stage": STAGE,
        "run_prefix": RUN_PREFIX,
        "scenario_name": SCENARIO_NAME,
        "base_chkpt_parent": BASE_CHKPT_DIR,          # tmp_avoid_dynamic/maddpgwithatt_origin3D
        "chkpt_dir_with_sep": CHKPT_DIR,              # 末尾带分隔符
        "model_save_dir": MODEL_SAVE_DIR,             # 完整模型目录路径
        "log_dir": LOG_DIR,                           # log_arvp_origin3D/前缀_arvp_origin3D
        "runs_parent_dir": RUNS_PARENT_DIR,           # runs/前缀_arvp_origin3D
        "reward_png_name": f"{RUN_PREFIX}_{STAGE}_arvp_origin3D_reward.png",
        # --- 模型维度（evaluate 端要匹配）---
        "obs_agt_dim": 56,
        "obs_tar_dim": 50,
        "n_agents": 4,
        "n_actions": 3,
        # --- 训练标识 ---
        "method": "origin3D: 原生 3D MADDPG-LSTM-Att-PER 基线",
        "created_at": datetime.now().isoformat(timespec='seconds'),
    }
    cfg_path = save_run_config(RUN_CONFIG)
    print(f"[origin3D-基线] 运行参数配置（供 evaluate 读取）已保存: {cfg_path}")

    env = UAVEnv()
    log_dir_tb = create_unique_log_dir(RUNS_UAV_BASE)
    writer = SummaryWriter(log_dir=log_dir_tb)
    print(f"[origin3D-基线] TensorBoard runs 目录: {log_dir_tb}/")

    n_agents = env.num_agents
    actor_dims = []
    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)

    print(f"[origin3D-基线] actor_dims = {actor_dims}, critic_dims = {critic_dims}")

    n_actions = 3  # 3D: [ax, ay, az]
    alpha = 0.00001
    beta = 0.001

    # 3D: obs_agt=56, obs_tar=50
    obs_agt = actor_dims[0]
    obs_tar = actor_dims[-1]
    maddpg_agents = MADDPGWithAttention(actor_dims, critic_dims, n_agents, n_actions,
                                        alpha=alpha, beta=beta, scenario=SCENARIO_NAME,
                                        chkpt_dir=CHKPT_DIR,
                                        obs_agt=obs_agt, obs_tar=obs_tar)

    # 保存超参数文本（放到模型目录）
    hyperparams_txt_path = os.path.join(MODEL_SAVE_DIR, 'hyperparameters.txt')
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    with open(hyperparams_txt_path, 'w') as f:
        f.write(f'run_prefix: {RUN_PREFIX}\n')
        f.write(f'stage: {STAGE}\n')
        f.write(f'alpha: {alpha}\n')
        f.write(f'beta: {beta}\n')
        f.write(f'Obs dim agent: {obs_agt}, target: {obs_tar}\n')
        f.write(f'n_actions: {n_actions} (3D: ax, ay, az)\n')
        f.write(f'reward components: mu1=0.9, mu2=0.2, mu3=0.0(stage1), mu4=10, mu5=0.1, mu6=0.5, mu7=0.1\n')
    print(f"[origin3D-基线] 超参数文本保存到: {hyperparams_txt_path}")

    memory = PERMultiAgentReplayBuffer(1000000, critic_dims, actor_dims,
                                       n_actions, n_agents, batch_size=256)

    BATCH_SIZE = 100
    N_GAMES = 5000
    MAX_STEPS = 110
    total_steps = 0
    score_history = []
    target_score_history = []
    evaluate = False
    best_score = -1000
    window_size = 100
    success_evaluator_each = []
    print(f"[origin3D-基线] checkpoint 模型保存目录: {MODEL_SAVE_DIR}/")
    print(f"[origin3D-基线] 第一阶段训练 (mu3=0.0): N_GAMES={N_GAMES}")

    if evaluate:
        try:
            maddpg_agents.load_checkpoint()
        except Exception as e:
            print(f'[warn] load_checkpoint skipped (no/incompatible checkpoint): {e}')
        print('----  evaluating  ----')
    else:
        print('----training (origin3D baseline) start----')

    evaluator = UAVTaskEvaluator([0, 0, 0], max_time_steps=MAX_STEPS, dones=[False, False, False, False])

    i = 0
    while i < N_GAMES:
        with tqdm(total=BATCH_SIZE, desc=f"Batch Progress (Total {i}/{N_GAMES}) [origin3D][{STAGE}]", unit="episode") as pbar:
            for _ in range(BATCH_SIZE):
                if i >= N_GAMES:
                    break

                obs = env.reset()
                # 校验观测维度是否正确
                if i == 0:
                    for agt_idx, o in enumerate(obs):
                        print(f"  [DEBUG] agent_{agt_idx} obs.shape = {len(o)}")
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

                    if total_steps % 10 == 0 and not evaluate:
                        maddpg_agents.learn(memory, total_steps, state)

                    obs = obs_
                    score += sum(rewards[0:3])
                    score_target += rewards[-1]
                    total_steps += 1
                    episode_step += 1

                if success_evaluator_total and episode_step < MAX_STEPS:
                    print(f"第 {i} 回合，总的成功率: {success_evaluator_total * 100:.2f}%")
                score_history.append(score)
                target_score_history.append(score_target)
                writer.add_scalar("UAV Reward (origin3D)", score, i)
                writer.add_scalar("Target Reward (origin3D)", score_target, i)

                pbar.update(1)
                pbar.set_postfix({
                    "Episode Reward": f"{score:.2f}",
                    "Target Reward": f"{score_target:.2f}"
                })
                i += 1

            avg_score = np.mean(score_history[-BATCH_SIZE:])
            avg_target_score = np.mean(target_score_history[-BATCH_SIZE:])

            pbar.close()
            tqdm.write(f'Batch {i // BATCH_SIZE} completed (origin3D).')
            tqdm.write(f'Average Score: {avg_score:.2f}, Average Target Score: {avg_target_score:.2f}')

            if avg_score > best_score:
                tqdm.write(f'New best avg score {avg_score:.2f} > previous best score {best_score:.2f}. Saving models...')
                maddpg_agents.save_checkpoint()
                best_score = avg_score

    writer.close()

    # ============ 训练结束：保存结果到 LOG_DIR ============
    # 1) 保存奖励历史 CSV
    csv_path = os.path.join(LOG_DIR, 'score_history.csv')
    df = pd.DataFrame({
        'episode': list(range(len(score_history))),
        'total_reward': score_history,
        'target_reward': target_score_history,
    })
    # 滑动均值列
    if len(score_history) >= window_size:
        sm = moving_average(score_history, window_size)
        padded_sm = [np.nan] * (window_size - 1) + list(sm)
        df['smoothed_total_reward_100'] = padded_sm
        if len(target_score_history) >= window_size:
            sm_t = moving_average(target_score_history, window_size)
            padded_sm_t = [np.nan] * (window_size - 1) + list(sm_t)
            df['smoothed_target_reward_100'] = padded_sm_t
    df.to_csv(csv_path, index=False)
    print(f"[origin3D-基线] 奖励历史 CSV 保存到: {csv_path}")

    # 2) 保存超参数 CSV
    hp_csv = os.path.join(LOG_DIR, 'hyperparameters.csv')
    hp_rows = [
        ('N_GAMES', N_GAMES),
        ('MAX_STEPS', MAX_STEPS),
        ('BATCH_SIZE_memory', 256),
        ('BATCH_SIZE_print', BATCH_SIZE),
        ('alpha', alpha),
        ('beta', beta),
        ('replay_buffer', 'PERMultiAgentReplayBuffer'),
        ('buffer_size', 1000000),
        ('obs_agt_dim', obs_agt),
        ('obs_tar_dim', obs_tar),
        ('n_actions', n_actions),
        ('mu1_r_near', 0.9),
        ('mu2_r_safe', 0.2),
        ('mu3_r_stage', 0.0),
        ('mu4_r_finish', 10),
        ('mu5_vo', 0.1),
        ('mu6_shaping', 0.5),
        ('mu7_align', 0.1),
        ('method', 'origin3D: 原生 3D MADDPG-LSTM-Att-PER 基线'),
        ('stage', STAGE),
        ('run_prefix', RUN_PREFIX),
        ('total_trained_episodes', len(score_history)),
    ]
    pd.DataFrame(hp_rows, columns=['parameter', 'value']).to_csv(hp_csv, index=False)
    print(f"[origin3D-基线] 超参数 CSV 保存到: {hp_csv}")

    # 3) 绘制并保存 reward 曲线（三处）
    #    - 文件名：当前日期+时间_stage参数_arvp_origin3D_reward.png
    #    - 保存位置：1) 模型保存目录 MODEL_SAVE_DIR  2) LOG_DIR  3) 工作目录根
    REWARD_PNG_NAME = f"{RUN_PREFIX}_{STAGE}_arvp_origin3D_reward.png"

    plt.figure(figsize=(12, 6))
    plt.plot(score_history, label='Raw Total Reward', alpha=0.3)
    if len(score_history) >= window_size:
        score_history_ma = moving_average(score_history, window_size)
        plt.plot(range(window_size - 1, len(score_history)), score_history_ma,
                 label=f'Smoothed Total Reward (win={window_size})', linewidth=2)
    plt.title(f'ARVP-MADDPG origin3D baseline - {RUN_PREFIX}_{STAGE} - N={len(score_history)}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存三处
    root_png = REWARD_PNG_NAME  # 工作目录根
    model_png = os.path.join(MODEL_SAVE_DIR, REWARD_PNG_NAME)  # 模型保存目录
    log_png = os.path.join(LOG_DIR, REWARD_PNG_NAME)  # LOG目录
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    plt.savefig(root_png, dpi=150)
    plt.savefig(model_png, dpi=150)
    plt.savefig(log_png, dpi=150)
    plt.close()
    print(f"[origin3D-基线] reward 曲线保存到:\n  - {os.path.abspath(root_png)}\n  - {os.path.abspath(model_png)}\n  - {os.path.abspath(log_png)}")

    print(f"\n========== origin3D 基线训练完成 ==========")
    print(f"RUN_PREFIX: {RUN_PREFIX}")
    print(f"STAGE: {STAGE}")
    print(f"模型目录: {MODEL_SAVE_DIR}/")
    print(f"日志目录: {LOG_DIR}/")
    print(f"TensorBoard runs 目录: {log_dir_tb}/")
    print(f"总回合数: {len(score_history)}")
    print(f"最后 100 回合平均奖励: {np.mean(score_history[-100:]):.3f}")
    print(f"历史最佳平均奖励: {best_score:.3f}")
