"""
train_arvp2D.py
================
基于 train_arvp.py 创建的 2D 追逃训练脚本，采用“日期时间+stage”前缀的目录组织方式
（参考方案一 APF 训练脚本的目录管理风格）。

关键设计（路径中心化）：
  - 所有相关文件夹路径（模型 / 日志 / TensorBoard runs / 配置 JSON）只在本文件指定一次。
  - 其它相关文件（model/maddpg_att.py、evaluate_arvp.py 等）只接收本文件传出的路径参数，
    不再硬编码任何目录。
  - 本文件把路径/超参写入 _last_arvp2D_run.json，子进程 evaluate_arvp.py 通过 --from-config
    读取该 JSON，从而加载正确的模型检查点路径。

与 train_arvp.py 的差异：
  - 观测维度保持基线 obs_agt=26, obs_tar=23（不引入 APF 增强）。
  - 从零开始训练（不再 load_checkpoint，因为新分支已清空旧模型）。
  - N_GAMES=5000。
  - 自动创建 log_arvp2D/ 目录保存 reward PNG + score_history.csv + hyperparams.csv。
  - 模型保存在 tmp_avoid_dynamic/maddpgwithatt_2D/{RUN_PREFIX}_UAV_Round_up。
  - runs/ 下创建带前缀的子目录存放 TensorBoard 事件。
  - 手动/默认 stage 参数，目录使用 “当前日期+时间_stage参数_” 前缀。
"""
import numpy as np
from sympy import evaluate  # noqa: F401  (与 train_arvp.py 保持一致的无副作用导入)

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


RUN_CONFIG_FILENAME = "_last_arvp2D_run.json"  # 参数传递文件（根目录下，供 evaluate_arvp2D 读取）


def save_run_config(config: dict) -> str:
    """把本次训练的关键路径/参数写到 JSON，供 evaluate_arvp.py 直接读取（参数中心化）"""
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
    # ================== stage 参数 ==================
    DEFAULT_STAGE = "stage1"
    # user_input = input(f"请输入 stage 参数（直接回车使用默认值 '{DEFAULT_STAGE}'）: ").strip()
    # STAGE = user_input if user_input else DEFAULT_STAGE
    STAGE = DEFAULT_STAGE
    print(f"[train_arvp2D] 使用 stage = {STAGE}")

    # 生成日期时间+stage 前缀（全局唯一标识）
    RUN_PREFIX = get_run_prefix(STAGE)
    print(f"[train_arvp2D] RUN_PREFIX = {RUN_PREFIX}")

    # ================== runs/ 下创建带前缀的目录，UAV_trainingXX 放其中 ==================
    RUNS_PARENT_DIR = os.path.join("runs", f"{RUN_PREFIX}_arvp2D")
    os.makedirs(RUNS_PARENT_DIR, exist_ok=True)
    RUNS_UAV_BASE = os.path.join(RUNS_PARENT_DIR, "UAV_training")

    # ================== log_arvp2D/ 下创建带前缀的目录 ==================
    LOG_DIR = os.path.join("log_arvp2D", f"{RUN_PREFIX}_arvp2D")
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"[train_arvp2D] 日志目录: {LOG_DIR}/")

    # ================== 模型保存目录：tmp_avoid_dynamic/maddpgwithatt_2D/{RUN_PREFIX}_UAV_Round_up ==================
    BASE_CHKPT_DIR = os.path.join("tmp_avoid_dynamic", "maddpgwithatt_2D")
    os.makedirs(BASE_CHKPT_DIR, exist_ok=True)
    # scenario 会拼在 chkpt_dir 后面，最终模型目录 = BASE_CHKPT_DIR / {RUN_PREFIX}_UAV_Round_up
    CHKPT_DIR = BASE_CHKPT_DIR + os.sep  # 结尾加分隔符，与旧代码保持一致
    SCENARIO_NAME = f"{RUN_PREFIX}_UAV_Round_up"
    MODEL_SAVE_DIR = os.path.join(BASE_CHKPT_DIR, SCENARIO_NAME)  # 用于日志打印

    # ================== 参数中心化：保存 JSON 配置供 evaluate_arvp.py 直接读取 ==================
    RUN_CONFIG = {
        # --- 路径相关 ---
        "stage": STAGE,
        "run_prefix": RUN_PREFIX,
        "scenario_name": SCENARIO_NAME,
        "base_chkpt_parent": BASE_CHKPT_DIR,          # tmp_avoid_dynamic/maddpgwithatt_2D
        "chkpt_dir_with_sep": CHKPT_DIR,              # 末尾带分隔符
        "model_save_dir": MODEL_SAVE_DIR,             # 完整模型目录路径
        "log_dir": LOG_DIR,                           # log_arvp2D/前缀_arvp2D
        "runs_parent_dir": RUNS_PARENT_DIR,           # runs/前缀_arvp2D
        "reward_png_name": f"{RUN_PREFIX}_{STAGE}_arvp2D_reward.png",
        "config_filename": RUN_CONFIG_FILENAME,
        # --- 模型维度（evaluate 端要匹配）---
        "obs_agt_dim": 26,
        "obs_tar_dim": 23,
        "n_agents": 4,
        "n_actions": 2,
        # --- 训练标识 ---
        "method": "train_arvp2D: 基于 train_arvp.py 的 MADDPG+Attention 基线训练 (2D)",
        "created_at": datetime.now().isoformat(timespec='seconds'),
    }
    cfg_path = save_run_config(RUN_CONFIG)
    print(f"[train_arvp2D] 运行参数配置（供 evaluate 读取）已保存: {cfg_path}")

    env = UAVEnv()
    log_dir_tb = create_unique_log_dir(RUNS_UAV_BASE)
    writer = SummaryWriter(log_dir=log_dir_tb)
    print(f"[train_arvp2D] TensorBoard runs 目录: {log_dir_tb}/")

    n_agents = env.num_agents
    actor_dims = []
    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)

    print(f"[train_arvp2D] actor_dims = {actor_dims}, critic_dims = {critic_dims}")

    n_actions = 2
    alpha = 0.00001
    beta = 0.001

    # 基线版本：观测维度 obs_agt=26, obs_tar=23（与 train_arvp.py 一致）
    maddpg_agents = MADDPGWithAttention(actor_dims, critic_dims, n_agents, n_actions,
                                        alpha=alpha, beta=beta, scenario=SCENARIO_NAME,
                                        chkpt_dir=CHKPT_DIR,
                                        obs_agt=26, obs_tar=23)

    # 保存超参数文本（放到模型目录）
    hyperparams_txt_path = os.path.join(MODEL_SAVE_DIR, 'hyperparameters.txt')
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    with open(hyperparams_txt_path, 'w') as f:
        f.write(f'run_prefix: {RUN_PREFIX}\n')
        f.write(f'stage: {STAGE}\n')
        f.write(f'alpha: {alpha}\n')
        f.write(f'beta: {beta}\n')
        f.write(f'Obs dim agent: 26, target: 23\n')
        f.write(f'method: {RUN_CONFIG["method"]}\n')
    print(f"[train_arvp2D] 超参数文本保存到: {hyperparams_txt_path}")

    memory = PERMultiAgentReplayBuffer(1000000, critic_dims, actor_dims,
                                       n_actions, n_agents, batch_size=256)

    BATCH_SIZE = 100
    N_GAMES = 5000  # 严格 5000 episode
    MAX_STEPS = 110
    total_steps = 0
    score_history = []
    target_score_history = []
    evaluate = False
    best_score = -1000
    window_size = 100
    success_evaluator_each = []
    print(f"[train_arvp2D] checkpoint 模型保存目录: {MODEL_SAVE_DIR}/")
    print(f"[train_arvp2D] 训练 (2D baseline): N_GAMES={N_GAMES}")
    # 新分支已清空旧模型，从零训练，不加载旧权重
    # maddpg_agents.load_checkpoint()

    if evaluate:
        maddpg_agents.load_checkpoint()
        print('----  evaluating  ----')
    else:
        print('----training (2D baseline) start----')

    evaluator = UAVTaskEvaluator([0, 0, 0], max_time_steps=MAX_STEPS, dones=[False, False, False, False])

    i = 0
    while i < N_GAMES:
        with tqdm(total=BATCH_SIZE, desc=f"Batch Progress (Total {i}/{N_GAMES}) [2D][{STAGE}]", unit="episode") as pbar:
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
                writer.add_scalar("UAV Reward (2D)", score, i)
                writer.add_scalar("Target Reward (2D)", score_target, i)

                pbar.update(1)
                pbar.set_postfix({
                    "Episode Reward": f"{score:.2f}",
                    "Target Reward": f"{score_target:.2f}"
                })
                i += 1

            avg_score = np.mean(score_history[-BATCH_SIZE:])
            avg_target_score = np.mean(target_score_history[-BATCH_SIZE:])

            pbar.close()
            tqdm.write(f'Batch {i // BATCH_SIZE} completed (2D).')
            tqdm.write(f'Average Score: {avg_score:.2f}, Average Target Score: {avg_target_score:.2f}')

            if avg_score > best_score:
                tqdm.write(f'New best avg score {avg_score:.2f} > previous best score {best_score:.2f}. Saving models...')
                maddpg_agents.save_checkpoint()
                best_score = avg_score
                # 调用 evaluate_arvp.py（参数通过 _last_arvp2D_run.json 自动读取，加载正确路径）
                # 使用 --headless --max-frames 50 做快速评估（跑完自动退出，无需 sleep 等待）
                try:
                    eval_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluate_arvp.py')
                    eval_cmd = [
                        sys.executable, eval_script,   # 使用训练时同一个 Python 解释器（避免 numpy/torch 环境不匹配）
                        '--headless',          # 不弹 GUI，跑完就退出
                        '--from-config',       # 优先从 _last_arvp2D_run.json 读取路径参数
                        '--max-frames', '50',  # 快速评估：最多 50 步
                    ]
                    tqdm.write(f'调用评估脚本: {" ".join(eval_cmd)}')
                    process = subprocess.Popen(
                        eval_cmd,
                        cwd=os.path.dirname(os.path.abspath(__file__)),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding='utf-8',       # 评估脚本输出含 ✓ 等非 GBK 字符，必须用 utf-8 解码
                        errors='replace',       # 万一有无法解码的字节也不崩溃
                    )
                    try:
                        stdout_data, _ = process.communicate(timeout=120)  # 最多等 2 分钟
                        if stdout_data:
                            # 输出评估日志到 tqdm write（避免打断进度条）
                            for line in stdout_data.strip().splitlines()[-30:]:
                                tqdm.write(f"  [eval] {line}")
                        tqdm.write(f"evaluate_arvp.py 完成，exit code={process.returncode}")
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                        tqdm.write("evaluate 超时（>120s），已终止。")
                except Exception as e:
                    tqdm.write(f"evaluate 调用失败（不影响训练继续）: {type(e).__name__}: {e}")

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
    print(f"[train_arvp2D] 奖励历史 CSV 保存到: {csv_path}")

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
        ('obs_agt_dim', 26),
        ('obs_tar_dim', 23),
        ('method', RUN_CONFIG["method"]),
        ('stage', STAGE),
        ('run_prefix', RUN_PREFIX),
        ('total_trained_episodes', len(score_history)),
    ]
    pd.DataFrame(hp_rows, columns=['parameter', 'value']).to_csv(hp_csv, index=False)
    print(f"[train_arvp2D] 超参数 CSV 保存到: {hp_csv}")

    # 3) 绘制并保存 reward 曲线
    #    - 文件名：当前日期+时间_stage参数_arvp2D_reward.png
    #    - 保存位置：1) 模型保存目录 MODEL_SAVE_DIR  2) LOG_DIR  3) 工作目录根
    REWARD_PNG_NAME = f"{RUN_PREFIX}_{STAGE}_arvp2D_reward.png"

    plt.figure(figsize=(12, 6))
    plt.plot(score_history, label='Raw Total Reward', alpha=0.3)
    if len(score_history) >= window_size:
        score_history_ma = moving_average(score_history, window_size)
        plt.plot(range(window_size - 1, len(score_history)), score_history_ma,
                 label=f'Smoothed Total Reward (win={window_size})', linewidth=2)
    plt.title(f'ARVP-MADDPG 2D Baseline - {RUN_PREFIX}_{STAGE} - N={len(score_history)}')
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
    print(f"[train_arvp2D] reward 曲线保存到:\n  - {os.path.abspath(root_png)}\n  - {os.path.abspath(model_png)}\n  - {os.path.abspath(log_png)}")

    print(f"\n========== train_arvp2D 训练完成 ==========")
    print(f"RUN_PREFIX: {RUN_PREFIX}")
    print(f"STAGE: {STAGE}")
    print(f"模型目录: {MODEL_SAVE_DIR}/")
    print(f"日志目录: {LOG_DIR}/")
    print(f"TensorBoard runs 目录: {log_dir_tb}/")
    print(f"总回合数: {len(score_history)}")
    print(f"最后 100 回合平均奖励: {np.mean(score_history[-100:]):.3f}")
    print(f"历史最佳平均奖励: {best_score:.3f}")
