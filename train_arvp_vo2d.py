import numpy as np
from sympy import evaluate

from model.maddpg_att import MADDPGWithAttention
from env.sim_env_vo2d import UAVEnv
from buffer.buffer import MultiAgentReplayBuffer
from buffer.PERbuffer import PERMultiAgentReplayBuffer
import time
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
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


RUN_CONFIG_FILENAME = "_last_arvp_vo2d_run.json"


def save_run_config(config: dict) -> str:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), RUN_CONFIG_FILENAME)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return path


warnings.filterwarnings('ignore')


def get_run_prefix(stage: str) -> str:
    now = datetime.now()
    ts = now.strftime("%m%d_%H%M%S")
    return f"{ts}_{stage}"


def create_unique_log_dir(base_dir: str) -> str:
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
    DEFAULT_STAGE = "stage1"
    STAGE = DEFAULT_STAGE
    print(f"[方案三-VO2D] 使用 stage = {STAGE}")

    RUN_PREFIX = get_run_prefix(STAGE)
    print(f"[方案三-VO2D] RUN_PREFIX = {RUN_PREFIX}")

    # ================== runs/ 下创建带前缀目录 ==================
    RUNS_PARENT_DIR = os.path.join("runs", f"{RUN_PREFIX}_arvp_vo2d")
    os.makedirs(RUNS_PARENT_DIR, exist_ok=True)
    RUNS_UAV_BASE = os.path.join(RUNS_PARENT_DIR, "UAV_training")

    # ================== log_arvp_vo2d/ 下创建带前缀目录 ==================
    LOG_DIR = os.path.join("log_arvp_vo2d", f"{RUN_PREFIX}_arvp_vo2d")
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"[方案三-VO2D] 日志目录: {LOG_DIR}/")

    # ================== 模型保存目录 ==================
    BASE_CHKPT_DIR = os.path.join("tmp_avoid_dynamic", "maddpgwithatt_vo2d")
    os.makedirs(BASE_CHKPT_DIR, exist_ok=True)
    CHKPT_DIR = BASE_CHKPT_DIR + os.sep
    SCENARIO_NAME = f"{RUN_PREFIX}_UAV_Round_up"
    MODEL_SAVE_DIR = os.path.join(BASE_CHKPT_DIR, SCENARIO_NAME)

    # ================== 参数中心化 JSON 配置 ==================
    RUN_CONFIG = {
        "stage": STAGE,
        "run_prefix": RUN_PREFIX,
        "scenario_name": SCENARIO_NAME,
        "base_chkpt_parent": BASE_CHKPT_DIR,
        "chkpt_dir_with_sep": CHKPT_DIR,
        "model_save_dir": MODEL_SAVE_DIR,
        "log_dir": LOG_DIR,
        "runs_parent_dir": RUNS_PARENT_DIR,
        "reward_png_name": f"{RUN_PREFIX}_{STAGE}_arvp_vo2d_reward.png",
        # --- 方案三 VO2D 维度 ---
        "obs_agt_dim": 32,
        "obs_tar_dim": 29,
        "n_agents": 4,
        "n_actions": 2,
        "method": "方案三: VO速度锥约束的动作空间投影 + 2D立体锥观测 (VO2D)",
        "created_at": datetime.now().isoformat(timespec='seconds'),
    }
    cfg_path = save_run_config(RUN_CONFIG)
    print(f"[方案三-VO2D] 运行参数配置（供 evaluate 读取）已保存: {cfg_path}")

    env = UAVEnv()
    log_dir_tb = create_unique_log_dir(RUNS_UAV_BASE)
    writer = SummaryWriter(log_dir=log_dir_tb)
    print(f"[方案三-VO2D] TensorBoard runs 目录: {log_dir_tb}/")

    n_agents = env.num_agents
    actor_dims = []
    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)

    print(f"[方案三-VO2D] actor_dims = {actor_dims}, critic_dims = {critic_dims}")

    n_actions = 2
    alpha = 0.00001
    beta = 0.001

    # VO2D: obs_agt=32, obs_tar=29
    maddpg_agents = MADDPGWithAttention(actor_dims, critic_dims, n_agents, n_actions,
                                        alpha=alpha, beta=beta, scenario=SCENARIO_NAME,
                                        chkpt_dir=CHKPT_DIR,
                                        obs_agt=32, obs_tar=29)

    hyperparams_txt_path = os.path.join(MODEL_SAVE_DIR, 'hyperparameters.txt')
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    with open(hyperparams_txt_path, 'w') as f:
        f.write(f'run_prefix: {RUN_PREFIX}\n')
        f.write(f'stage: {STAGE}\n')
        f.write(f'alpha: {alpha}\n')
        f.write(f'beta: {beta}\n')
        f.write(f'VO mu8 (修正惩罚系数): {env.mu8}\n')
        f.write(f'VO mu9 (预测避碰加分系数): {env.mu9}\n')
        f.write(f'VO TTC_threshold (s): {env.TTC_threshold}\n')
        f.write(f'VO prox_threshold (距离阈值): {env.vo_prox_threshold}\n')
        f.write(f'Obs dim agent: 32, target: 29\n')
        f.write(f'Obs VO2D features per agent: 6 (invasion_score, min_ttc_norm, active_count, escape_x, escape_y, total_occ_angle)\n')
        f.write(f'reward components: mu1=0.9, mu2=0.2, mu3=0.0(stage1), mu4=10, mu5=0.1, mu8={env.mu8}, mu9={env.mu9}\n')
        f.write(f'创新点1: 2D VO 立体锥观测编码 (6-dim compact features)\n')
        f.write(f'创新点2: Post-Action VO Projection 层 (动作执行前投影到VO可行域)\n')
        f.write(f'创新点3: mu8修正惩罚 + mu9预测性避碰加分\n')
    print(f"[方案三-VO2D] 超参数文本保存到: {hyperparams_txt_path}")

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
    print(f"[方案三-VO2D] checkpoint 模型保存目录: {MODEL_SAVE_DIR}/")
    print(f"[方案三-VO2D] 第一阶段训练 (mu3=0.0): N_GAMES={N_GAMES}")

    if STAGE == "stage2":
        maddpg_agents.load_checkpoint()

    if evaluate:
        maddpg_agents.load_checkpoint()
        print('----  evaluating  ----')
    else:
        print('----training (VO2D-Augmented) start----')

    evaluator = UAVTaskEvaluator([0, 0, 0], max_time_steps=MAX_STEPS, dones=[False, False, False, False])

    i = 0
    while i < N_GAMES:
        with tqdm(total=BATCH_SIZE, desc=f"Batch Progress (Total {i}/{N_GAMES}) [VO2D][{STAGE}]", unit="episode") as pbar:
            for _ in range(BATCH_SIZE):
                if i >= N_GAMES:
                    break

                obs = env.reset()
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
                writer.add_scalar("UAV Reward (VO2D)", score, i)
                writer.add_scalar("Target Reward (VO2D)", score_target, i)

                pbar.update(1)
                pbar.set_postfix({
                    "Episode Reward": f"{score:.2f}",
                    "Target Reward": f"{score_target:.2f}"
                })
                i += 1

            avg_score = np.mean(score_history[-BATCH_SIZE:])
            avg_target_score = np.mean(target_score_history[-BATCH_SIZE:])

            pbar.close()
            tqdm.write(f'Batch {i // BATCH_SIZE} completed (VO2D).')
            tqdm.write(f'Average Score: {avg_score:.2f}, Average Target Score: {avg_target_score:.2f}')

            if avg_score > best_score:
                tqdm.write(f'New best avg score {avg_score:.2f} > previous best score {best_score:.2f}. Saving models...')
                maddpg_agents.save_checkpoint()
                best_score = avg_score
                try:
                    eval_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluate_arvp_vo2d.py')
                    eval_cmd = [
                        sys.executable, eval_script,
                        '--headless',
                        '--from-config',
                        '--max-frames', '50',
                    ]
                    tqdm.write(f'调用评估脚本: {" ".join(eval_cmd)}')
                    process = subprocess.Popen(
                        eval_cmd,
                        cwd=os.path.dirname(os.path.abspath(__file__)),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                    )
                    try:
                        stdout_data, _ = process.communicate(timeout=120)
                        if stdout_data:
                            for line in stdout_data.strip().splitlines()[-30:]:
                                tqdm.write(f"  [eval] {line}")
                        tqdm.write(f"evaluate_arvp_vo2d.py 完成，exit code={process.returncode}")
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                        tqdm.write("evaluate 超时（>120s），已终止。")
                except Exception as e:
                    tqdm.write(f"evaluate 调用失败（不影响训练继续）: {type(e).__name__}: {e}")

    writer.close()

    # ============ 训练结束：保存结果 ============
    csv_path = os.path.join(LOG_DIR, 'score_history.csv')
    df = pd.DataFrame({
        'episode': list(range(len(score_history))),
        'total_reward': score_history,
        'target_reward': target_score_history,
    })
    if len(score_history) >= window_size:
        sm = moving_average(score_history, window_size)
        padded_sm = [np.nan] * (window_size - 1) + list(sm)
        df['smoothed_total_reward_100'] = padded_sm
        if len(target_score_history) >= window_size:
            sm_t = moving_average(target_score_history, window_size)
            padded_sm_t = [np.nan] * (window_size - 1) + list(sm_t)
            df['smoothed_target_reward_100'] = padded_sm_t
    df.to_csv(csv_path, index=False)
    print(f"[方案三-VO2D] 奖励历史 CSV 保存到: {csv_path}")

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
        ('obs_agt_dim', 32),
        ('obs_tar_dim', 29),
        ('mu1_r_near', 0.9),
        ('mu2_r_safe', 0.2),
        ('mu3_r_stage', 0.0),
        ('mu4_r_finish', 10),
        ('mu5_vo', 0.1),
        ('mu8_vo_correction_penalty', env.mu8),
        ('mu9_vo_predictive_avoid_bonus', env.mu9),
        ('vo_TTC_threshold_sec', env.TTC_threshold),
        ('vo_prox_threshold', env.vo_prox_threshold),
        ('method', '方案三: VO速度锥约束的动作空间投影 + 2D立体锥观测 (VO2D)'),
        ('stage', STAGE),
        ('run_prefix', RUN_PREFIX),
        ('total_trained_episodes', len(score_history)),
    ]
    pd.DataFrame(hp_rows, columns=['parameter', 'value']).to_csv(hp_csv, index=False)
    print(f"[方案三-VO2D] 超参数 CSV 保存到: {hp_csv}")

    REWARD_PNG_NAME = f"{RUN_PREFIX}_{STAGE}_arvp_vo2d_reward.png"

    plt.figure(figsize=(12, 6))
    plt.plot(score_history, label='Raw Total Reward', alpha=0.3)
    if len(score_history) >= window_size:
        score_history_ma = moving_average(score_history, window_size)
        plt.plot(range(window_size - 1, len(score_history)), score_history_ma,
                 label=f'Smoothed Total Reward (win={window_size})', linewidth=2)
    plt.title(f'ARVP-MADDPG + VO2D (方案三) - {RUN_PREFIX}_{STAGE} - N={len(score_history)}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    root_png = REWARD_PNG_NAME
    model_png = os.path.join(MODEL_SAVE_DIR, REWARD_PNG_NAME)
    log_png = os.path.join(LOG_DIR, REWARD_PNG_NAME)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    plt.savefig(root_png, dpi=150)
    plt.savefig(model_png, dpi=150)
    plt.savefig(log_png, dpi=150)
    plt.close()
    print(f"[方案三-VO2D] reward 曲线保存到:\n  - {os.path.abspath(root_png)}\n  - {os.path.abspath(model_png)}\n  - {os.path.abspath(log_png)}")

    print(f"\n========== 方案三 (VO2D) 训练完成 ==========")
    print(f"RUN_PREFIX: {RUN_PREFIX}")
    print(f"STAGE: {STAGE}")
    print(f"模型目录: {MODEL_SAVE_DIR}/")
    print(f"日志目录: {LOG_DIR}/")
    print(f"TensorBoard runs 目录: {log_dir_tb}/")
    print(f"总回合数: {len(score_history)}")
    print(f"最后 100 回合平均奖励: {np.mean(score_history[-100:]):.3f}")
    print(f"历史最佳平均奖励: {best_score:.3f}")
