"""
ARVP-APF 方案评估脚本
适配：train_arvp_apf.py 训练出的模型（obs_agt / obs_tar 维度、scenario 路径全部从
项目根目录的 _last_arvp_apf_run.json 中自动读取，无需在此硬编码，保证与训练 100% 一致）

优先级（避免冲突）：
  1. 命令行 --scenario / --chkpt-parent / --obs-dim 显式指定
  2. 命令行 --from-config 开关 或 检测到 _last_arvp_apf_run.json 存在 → 自动从 JSON 读
  3. 交互选择 scenario 目录

特点：
  1. 自动扫描可用的 scenario 目录，让用户选择（或默认选最新修改的）
  2. 支持 2D 可视化（与 evaluate_arvp.py 一致）和 3D 可视化模式
     - 2D：xy 平面轨迹 + VO 区域
     - 3D：在 z 维度增加高度层次（UAV / Target / Obstacle 分别在不同高度），
           方便直观观察合围关系；未来若环境升级到真实 3D，只需修改 z 来源即可
  3. evaluate=True：不加噪声，展示模型真实性能
  4. 结束时打印 UAVTaskEvaluator 综合得分（完成度、路径效率、碰撞、时间效率）

用法：
  python evaluate_arvp_apf.py                # 自动读 _last_arvp_apf_run.json → 默认 3D
  python evaluate_arvp_apf.py --2d           # 强制 2D 模式
  python evaluate_arvp_apf.py --headless     # 不显示 GUI，跑完就退出，保存结果
  python evaluate_arvp_apf.py --no-config    # 忽略 JSON 配置，强制交互选择
"""
import os
import sys
import time
import glob
import warnings
import argparse
import json
from datetime import datetime

import numpy as np
import matplotlib

warnings.filterwarnings('ignore')

RUN_CONFIG_FILENAME = "_last_arvp_apf_run.json"  # 与 train_arvp_apf.py 保持一致


def find_project_root() -> str:
    """定位项目根（根据 _last_arvp_apf_run.json 或 evaluate_arvp_apf.py 所在目录）"""
    here = os.path.dirname(os.path.abspath(__file__))
    # 优先 here 作为根（脚本就在根下）
    candidate = os.path.join(here, RUN_CONFIG_FILENAME)
    if os.path.isfile(candidate):
        return here
    # 向上搜索最多 3 层
    p = here
    for _ in range(3):
        p = os.path.dirname(p)
        if os.path.isfile(os.path.join(p, RUN_CONFIG_FILENAME)):
            return p
    return here


PROJECT_ROOT = find_project_root()


def load_run_config() -> dict:
    """读取 _last_arvp_apf_run.json；失败返回 {}"""
    path = os.path.join(PROJECT_ROOT, RUN_CONFIG_FILENAME)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        if isinstance(cfg, dict):
            return cfg
    except Exception as e:
        print(f"[WARN] 读取 {RUN_CONFIG_FILENAME} 失败: {e}，将回退到交互/默认值。")
    return {}


def parse_args():
    parser = argparse.ArgumentParser(description="ARVP-APF 评估脚本（支持 2D/3D 可视化 + 参数 JSON 自动读取）")
    parser.add_argument('--2d', dest='mode_2d', action='store_true',
                        help='使用 2D 可视化模式（默认 3D）')
    parser.add_argument('--headless', action='store_true',
                        help='无头模式：不显示 GUI 窗口，跑完保存截图后退出')
    parser.add_argument('--from-config', dest='from_config', action='store_true',
                        help=f'显式启用：从 {RUN_CONFIG_FILENAME} 读取 scenario / 路径 / 维度参数（默认行为，该开关为兼容保留）')
    parser.add_argument('--no-config', dest='no_config', action='store_true',
                        help=f'忽略 {RUN_CONFIG_FILENAME}，强制走命令行或交互选择（避免训练残留配置干扰）')
    parser.add_argument('--scenario', type=str, default=None,
                        help='指定 scenario 目录名（例如 0813_stage1_UAV_Round_up）；最高优先级')
    parser.add_argument('--chkpt-parent', type=str, default=None,
                        help='指定模型父目录（默认 tmp_avoid_dynamic/maddpgwithatt_apf）')
    parser.add_argument('--obs-agt-dim', type=int, default=None,
                        help='手动指定追捕者观测维度（默认读 JSON，或 32）')
    parser.add_argument('--obs-tar-dim', type=int, default=None,
                        help='手动指定目标观测维度（默认读 JSON，或 29）')
    parser.add_argument('--max-frames', type=int, default=300,
                        help='最大评估步数（默认 300）')
    parser.add_argument('--save-screenshot', type=str, default=None,
                        help='保存最终帧截图到指定路径（默认根据 scenario 自动命名）')
    return parser.parse_args()


ARGS = parse_args()
if ARGS.headless:
    matplotlib.use('Agg')  # 必须在 pyplot / animation 导入前设置

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # 3D 用

# ---------- 项目本地模块 ----------
from model.maddpg_att import MADDPGWithAttention
from env.sim_env_rvo import UAVEnv
from UAVTask import UAVTaskEvaluator


MODE_3D = not ARGS.mode_2d  # 默认 3D


def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


# ==================== 检查点目录发现 & 选择 ====================
DEFAULT_BASE_CHKPT_PARENT = os.path.join('tmp_avoid_dynamic', 'maddpgwithatt_apf')


def discover_scenarios(base_parent: str = None) -> list:
    """
    返回可用 scenario 目录列表（按修改时间倒序，最新在前）。
    base_parent=None 时使用 PROJECT_ROOT 下的默认 DEFAULT_BASE_CHKPT_PARENT。
    路径统一基于 PROJECT_ROOT 解析，避免从不同工作目录启动时找不到模型。
    """
    if base_parent is None:
        base_parent = os.path.join(PROJECT_ROOT, DEFAULT_BASE_CHKPT_PARENT)
    elif not os.path.isabs(base_parent):
        base_parent = os.path.join(PROJECT_ROOT, base_parent)

    if not os.path.isdir(base_parent):
        return []
    items = []
    for name in os.listdir(base_parent):
        full = os.path.join(base_parent, name)
        if not os.path.isdir(full):
            continue
        # 检查是否包含至少一个模型文件
        has_model = any(
            fn.endswith('_actor') or fn.endswith('_critic')
            for fn in os.listdir(full)
        )
        mtime = os.path.getmtime(full)
        items.append((mtime, name, full, has_model))
    items.sort(key=lambda x: x[0], reverse=True)  # 最新在前
    return items


def pick_scenario(base_parent: str, cli_scenario: str = None,
                  config_scenario: str = None) -> tuple:
    """
    返回 (scenario_name, chkpt_dir_with_sep, effective_base_parent)

    优先级：
      1. 命令行 --scenario
      2. JSON 配置 scenario_name
      3. 交互选择（无头模式下自动选最新修改的）
    """
    # 如果传了相对路径 base_parent（非绝对），基于 PROJECT_ROOT 解析
    if not os.path.isabs(base_parent):
        base_parent = os.path.join(PROJECT_ROOT, base_parent)

    # ====== 优先级 1：命令行 --scenario ======
    if cli_scenario is not None:
        full = os.path.join(base_parent, cli_scenario)
        if not os.path.isdir(full):
            print(f"[ERROR] --scenario 指定的目录不存在: {full}")
            print(f"        base_parent = {base_parent}")
            # 列出候选帮助用户定位
            cand = discover_scenarios(base_parent)
            if cand:
                print("        可用 scenario:")
                for _, n, f, hm in cand:
                    print(f"          - {n}{' [有模型]' if hm else ''}  ({f})")
            sys.exit(1)
        return cli_scenario, base_parent + os.sep, base_parent

    # ====== 优先级 2：JSON 配置 ======
    if config_scenario is not None:
        full = os.path.join(base_parent, config_scenario)
        if os.path.isdir(full):
            # 验证有模型
            has_model = any(
                fn.endswith('_actor') or fn.endswith('_critic')
                for fn in os.listdir(full)
            )
            if has_model:
                print(f"[INFO] 从 {RUN_CONFIG_FILENAME} 加载 scenario = {config_scenario}")
                return config_scenario, base_parent + os.sep, base_parent
            else:
                print(f"[WARN] JSON 指定 scenario={config_scenario} 下无模型文件，回退到交互选择。")
        else:
            print(f"[WARN] JSON 指定 scenario={config_scenario} 路径不存在 ({full})，回退到交互选择。")

    # ====== 优先级 3：交互 / 自动 ======
    scenarios = discover_scenarios(base_parent)
    if len(scenarios) == 0:
        print(f"[ERROR] 在 {base_parent}/ 下没有找到任何 scenario 目录。"
              f"请先运行 train_arvp_apf.py 训练模型，或用 --chkpt-parent 指定正确目录。")
        sys.exit(1)

    # 无头模式或非交互（TTY 不可用）→ 直接取最新
    headless_or_notty = ARGS.headless or (not sys.stdin.isatty())
    if headless_or_notty:
        mtime, name, full, has_model = scenarios[0]
        t = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[INFO] 自动选择最新 scenario: {name}  (修改时间 {t}){' [有模型]' if has_model else ''}")
        return name, base_parent + os.sep, base_parent

    print("\n==== 发现的可用模型 scenario（最新在前） ====")
    for i, (mtime, name, full, has_model) in enumerate(scenarios, 1):
        t = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        mark = " [有模型]" if has_model else " [目录空]"
        print(f"  [{i}] {name}  (修改时间 {t}){mark}")

    default_idx = 0
    try:
        user_in = input(f"\n请选择序号（直接回车使用默认 [{default_idx + 1}]，输入 q 退出）: ").strip()
    except EOFError:
        user_in = ''
    if user_in.lower() == 'q':
        print("用户取消。")
        sys.exit(0)
    if user_in == '':
        idx = default_idx
    else:
        try:
            idx = int(user_in) - 1
            if idx < 0 or idx >= len(scenarios):
                raise ValueError
        except ValueError:
            print(f"[ERROR] 无效输入: {user_in}")
            sys.exit(1)

    name = scenarios[idx][1]
    print(f"[INFO] 已选择 scenario: {name}")
    return name, base_parent + os.sep, base_parent


# ==================== 3D 辅助绘图函数 ====================
def make_3d_cylinder(x0, y0, radius, z_bottom, z_top, color='gray', alpha=0.5):
    """生成一个 3D 圆柱（障碍物），返回 Poly3DCollection"""
    theta = np.linspace(0, 2 * np.pi, 32)
    xs = x0 + radius * np.cos(theta)
    ys = y0 + radius * np.sin(theta)

    # 侧面
    verts_side = []
    for i in range(len(theta) - 1):
        quad = [
            (xs[i], ys[i], z_bottom),
            (xs[i + 1], ys[i + 1], z_bottom),
            (xs[i + 1], ys[i + 1], z_top),
            (xs[i], ys[i], z_top),
        ]
        verts_side.append(quad)

    # 顶面
    top_ring = [(xs[i], ys[i], z_top) for i in range(len(theta))]
    verts_cap = [top_ring]

    poly = Poly3DCollection(verts_side + verts_cap, facecolors=color, alpha=alpha, edgecolors='none')
    return poly


def plot_3d_frame(ax, env: UAVEnv, trajectories: list, velocities_magnitude: list):
    """
    绘制 3D 场景的一帧（真3D：使用环境中的真实 xyz 坐标）：
      - 立方体边界
      - 障碍物：3D 球体（线框）
      - 追捕者 UAV：3D scatter + 轨迹线
      - 目标：3D scatter + 轨迹线
    """
    ax.clear()

    L = env.length

    # ---- 立方体边界 ----
    corners = np.array([
        [0, 0, 0], [L, 0, 0], [L, L, 0], [0, L, 0], [0, 0, 0],
        [0, 0, L], [L, 0, L], [L, L, L], [0, L, L], [0, 0, L],
    ])
    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2],
            color='lightgray', linestyle='--', linewidth=0.8, alpha=0.7, label='Boundary')

    # ---- 障碍物（3D 球体线框）----
    for obs in env.obstacles:
        u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:6j]
        x = obs.position[0] + obs.radius * np.cos(u) * np.sin(v)
        y = obs.position[1] + obs.radius * np.sin(u) * np.sin(v)
        z = obs.position[2] + obs.radius * np.cos(v)
        ax.plot_wireframe(x, y, z, color='gray', alpha=0.3)

    # ---- 轨迹 + 当前位置（使用真实3D坐标）----
    COLORS = ['#1f77b4', '#2ca02c', '#9467bd', '#d62728']
    LABELS = ['UAV0 (Hunter)', 'UAV1 (Hunter)', 'UAV2 (Hunter)', 'Target']
    MARKERS = ['o', 'o', 'o', 's']

    for i in range(env.num_agents):
        traj = np.array(trajectories[i]) if trajectories[i] else np.zeros((0, 3))
        if len(traj) > 1:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                    color=COLORS[i], alpha=0.75, linewidth=1.5, label=LABELS[i])
        if len(traj) > 0:
            ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2],
                       color=COLORS[i], s=120, marker=MARKERS[i],
                       edgecolors='black', linewidths=0.5, depthshade=True)

    # ---- 坐标轴与视角 ----
    ax.set_xlim(-0.1, L + 0.1)
    ax.set_ylim(-0.1, L + 0.1)
    ax.set_zlim(-0.1, L + 0.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("ARVP-APF Multi-UAV Round-up [3D mode]")
    ax.legend(loc='upper left', fontsize=8)
    ax.view_init(elev=25, azim=-55)


# ==================== 2D 辅助绘图（保持与 evaluate_arvp.py 兼容） ====================
def plot_2d_frame(ax, env: UAVEnv, trajectories: list):
    ax.clear()
    L = env.length

    # 障碍物 (3D position -> XY projection)
    for obs in env.obstacles:
        circle = Circle((obs.position[0], obs.position[1]), obs.radius, color='gray', alpha=0.5)
        ax.add_patch(circle)

    COLORS = ['#1f77b4', '#2ca02c', '#9467bd', '#d62728']
    LABELS = ['UAV0', 'UAV1', 'UAV2', 'Target']

    for i in range(env.num_agents):
        traj = np.array(trajectories[i]) if trajectories[i] else np.zeros((0, 3))
        if len(traj) > 1:
            ax.plot(traj[:, 0], traj[:, 1], color=COLORS[i], alpha=0.7, label=LABELS[i])
        if len(traj) > 0:
            if i == env.num_agents - 1:
                ax.scatter(traj[-1, 0], traj[-1, 1], color=COLORS[i], s=80, marker='s',
                           edgecolors='black', label='Target')
            else:
                ax.scatter(traj[-1, 0], traj[-1, 1], color=COLORS[i], s=60,
                           edgecolors='black')

    ax.set_xlim(-0.1, L + 0.1)
    ax.set_ylim(-0.1, L + 0.1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("ARVP-APF Multi-UAV Round-up [2D mode]")
    ax.legend(loc='upper right', fontsize=8)


# ==================== 速度绘图（评估完弹出） ====================
def plot_velocities(velocities_magnitude, velocities_x, velocities_y, headless=False, save_path=None):
    window_size = 5
    smoothed_vm = [moving_average(v, window_size) if len(v) >= window_size else np.array(v)
                   for v in velocities_magnitude]
    smoothed_vx = [moving_average(v, window_size) if len(v) >= window_size else np.array(v)
                   for v in velocities_x]
    smoothed_vy = [moving_average(v, window_size) if len(v) >= window_size else np.array(v)
                   for v in velocities_y]
    n = min(len(smoothed_vm[0]), len(smoothed_vx[0]), len(smoothed_vy[0]))
    time_steps = list(range(n))

    fig, axs = plt.subplots(3, 1, figsize=(11, 9))
    COLORS = ['#1f77b4', '#2ca02c', '#9467bd', '#d62728']

    for i in range(len(smoothed_vm)):
        label = f'UAV{i}' if i != 3 else 'Target'
        axs[0].plot(time_steps, smoothed_vm[i][:n], color=COLORS[i], label=label)
    axs[0].set_title('Speed Magnitude vs Time (smoothed, win=5)')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('|v|')
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    for i in range(len(smoothed_vx)):
        label = f'UAV{i}' if i != 3 else 'Target'
        axs[1].plot(time_steps, smoothed_vx[i][:n], color=COLORS[i], label=label)
    axs[1].set_title('Velocity X Component vs Time')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('v_x')
    axs[1].legend()
    axs[1].grid(alpha=0.3)

    for i in range(len(smoothed_vy)):
        label = f'UAV{i}' if i != 3 else 'Target'
        axs[2].plot(time_steps, smoothed_vy[i][:n], color=COLORS[i], label=label)
    axs[2].set_title('Velocity Y Component vs Time')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('v_y')
    axs[2].legend()
    axs[2].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] 速度分析图已保存: {save_path}")
    if not headless:
        plt.show()
    else:
        plt.close(fig)


# ==================== 主评估流程 ====================
if __name__ == '__main__':
    print("=" * 60)
    print(" ARVP-APF 评估脚本  (支持 2D/3D 可视化 + JSON 参数自动读取)")
    print(f" 当前模式: {'2D' if ARGS.mode_2d else '3D'} | 无头模式: {ARGS.headless}")
    print(f" 项目根: {PROJECT_ROOT}")
    print("=" * 60)

    # ========== 加载 JSON 配置（除非 --no-config） ==========
    run_cfg = {}
    if not ARGS.no_config:
        run_cfg = load_run_config()
        if run_cfg:
            print(f"[INFO] 从 {RUN_CONFIG_FILENAME} 加载配置: "
                  f"stage={run_cfg.get('stage')}, run_prefix={run_cfg.get('run_prefix')}, "
                  f"scenario={run_cfg.get('scenario_name')}")
        else:
            if ARGS.from_config:
                print(f"[WARN] --from-config 启用，但 {RUN_CONFIG_FILENAME} 不存在/为空，将回退默认。")
    else:
        print(f"[INFO] --no-config 已指定，忽略 {RUN_CONFIG_FILENAME}。")

    # ========== 参数优先级合并 ==========
    # base_parent: --chkpt-parent > run_cfg.base_chkpt_parent > DEFAULT
    base_parent = (
        ARGS.chkpt_parent
        or run_cfg.get('base_chkpt_parent')
        or DEFAULT_BASE_CHKPT_PARENT
    )

    # obs 维度: 命令行 > run_cfg > 默认 32/29
    obs_agt_dim = (
        ARGS.obs_agt_dim
        if ARGS.obs_agt_dim is not None
        else int(run_cfg['obs_agt_dim'])
        if isinstance(run_cfg.get('obs_agt_dim'), (int, float)) and run_cfg['obs_agt_dim']
        else 56
    )
    obs_tar_dim = (
        ARGS.obs_tar_dim
        if ARGS.obs_tar_dim is not None
        else int(run_cfg['obs_tar_dim'])
        if isinstance(run_cfg.get('obs_tar_dim'), (int, float)) and run_cfg['obs_tar_dim']
        else 50
    )

    # 1. 选择 scenario 并构建加载路径（返回 3 元组）
    scenario_name, chkpt_dir, effective_base = pick_scenario(
        base_parent=base_parent,
        cli_scenario=ARGS.scenario,
        config_scenario=run_cfg.get('scenario_name'),
    )

    print(f"[INFO] chkpt_parent = {effective_base}")
    print(f"[INFO] scenario      = {scenario_name}")
    print(f"[INFO] obs 维度      = 追捕者 {obs_agt_dim}, 目标 {obs_tar_dim}")

    # 2. 初始化环境 + 模型
    env = UAVEnv()
    n_agents = env.num_agents
    n_actions = 3  # 3D: [ax, ay, az]

    actor_dims = []
    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)
    print(f"[INFO] actor_dims={actor_dims}, critic_dims={critic_dims}")

    # 校验观测维度与 JSON 一致（不一致给出 WARNING，避免静默失败）
    expected_agt = obs_agt_dim
    expected_tar = obs_tar_dim
    actual_agt = actor_dims[:-1]   # 除 target 外
    actual_tar = actor_dims[-1]
    for idx, actual in enumerate(actual_agt):
        if actual != expected_agt:
            print(f"[WARN] 追捕者 {idx} 实际观测维度 {actual} ≠ 配置期望 {expected_agt}，可能模型文件与环境不匹配！")
    if actual_tar != expected_tar:
        print(f"[WARN] Target 实际观测维度 {actual_tar} ≠ 配置期望 {expected_tar}，可能模型文件与环境不匹配！")

    maddpg_agents = MADDPGWithAttention(
        actor_dims, critic_dims, n_agents, n_actions,
        alpha=0.00001, beta=0.00001,  # 加载模型时 lr 不影响
        scenario=scenario_name,
        chkpt_dir=chkpt_dir,
        obs_agt=obs_agt_dim, obs_tar=obs_tar_dim,
    )

    # 3. 加载检查点
    print(f"[INFO] 正在加载模型: scenario={scenario_name}, chkpt_dir={chkpt_dir} ...")
    try:
        maddpg_agents.load_checkpoint()
    except Exception as e:
        print(f"[ERROR] 加载模型失败: {e}")
        print("  请确认目录下存在 agent_N_actor / agent_N_critic / target_* 等文件。")
        # 列出目录内容辅助排查
        model_dir = os.path.join(chkpt_dir.rstrip(os.sep), scenario_name)
        if os.path.isdir(model_dir):
            files = sorted(os.listdir(model_dir))
            if files:
                print(f"  目录 {model_dir} 下现有文件:")
                for fn in files[:30]:
                    print(f"    - {fn}")
            else:
                print(f"  目录 {model_dir} 为空！")
        else:
            print(f"  目录 {model_dir} 不存在！")
        sys.exit(1)
    print("---- Evaluating (evaluate=True, 无探索噪声) ----")

    # 4. 记录数组
    velocities_magnitude = [[] for _ in range(env.num_agents)]
    velocities_x = [[] for _ in range(env.num_agents)]
    velocities_y = [[] for _ in range(env.num_agents)]
    trajectories = [[] for _ in range(env.num_agents)]
    collisions_record = [[] for _ in range(env.num_agents)]

    # 5. 初始化第一次观测
    obs = env.reset()
    total_steps = 0
    finished = False
    finish_step = -1
    final_dones = [False] * n_agents
    final_pos = None

    # 6. 创建 Figure + Axes
    if MODE_3D:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=(8, 7))

    # 先画首帧，避免 FuncAnimation 第一次 update 前空窗
    for i in range(env.num_agents):
        trajectories[i].append(np.array(env.multi_current_pos[i], dtype=float).copy())
        vel = env.multi_current_vel[i]
        velocities_magnitude[i].append(float(np.linalg.norm(vel)))
        velocities_x[i].append(float(vel[0]))
        velocities_y[i].append(float(vel[1]))

    if MODE_3D:
        plot_3d_frame(ax, env, trajectories, velocities_magnitude)
    else:
        plot_2d_frame(ax, env, trajectories)

    # 7. update 函数
    def update(frame):
        global obs, total_steps, finished, finish_step, final_dones, final_pos

        if finished:
            return []

        # 选动作（evaluate=True，无噪声）
        actions = maddpg_agents.choose_action(obs, total_steps, evaluate=True)
        obs_, rewards, dones, collision_info, mul_pos = env.step(actions)

        for i in range(env.num_agents):
            trajectories[i].append(np.array(env.multi_current_pos[i], dtype=float).copy())
            collisions_record[i].append(collision_info)
            vel = env.multi_current_vel[i]
            velocities_magnitude[i].append(float(np.linalg.norm(vel)))
            velocities_x[i].append(float(vel[0]))
            velocities_y[i].append(float(vel[1]))

        if MODE_3D:
            plot_3d_frame(ax, env, trajectories, velocities_magnitude)
        else:
            plot_2d_frame(ax, env, trajectories)

        obs = obs_
        total_steps += 1

        # 结束判定
        if any(dones) or frame >= ARGS.max_frames:
            finished = True
            finish_step = frame
            final_dones = list(dones)
            final_pos = mul_pos

            success = all(dones) and frame < ARGS.max_frames
            status = "成功合围" if success else "超时 / 未完成"
            print(f"\n[评估结束] {status}！共用 {frame} 步。")
            print(f"  dones = {dones}")

            # 评测
            evaluator = UAVTaskEvaluator(
                target_position=[0, 0, 0],
                max_time_steps=ARGS.max_frames,
                dones=list(dones),
            )
            target_pos_final = env.multi_current_pos[-1]
            for i in range(env.num_agents):
                result = evaluator.evaluate(trajectories[i], collisions_record[i],
                                            target_pos_final)
                tag = f"UAV{i}" if i != env.num_agents - 1 else "Target"
                print(f"\n  [{tag}] 评测结果:")
                for k, v in result.items():
                    print(f"      {k:>22s}: {float(v):.3f}")

            # 保存最终截图（基于 PROJECT_ROOT 存放 eval_results_apf/）
            ss_path = ARGS.save_screenshot
            if ss_path is None:
                ts = datetime.now().strftime("%m%d_%H%M%S")
                suffix = "3d" if MODE_3D else "2d"
                prefix_tag = run_cfg.get('run_prefix', '') or 'eval'
                stage_tag = run_cfg.get('stage', '') or ''
                ss_dir = os.path.join(PROJECT_ROOT, "eval_results_apf", scenario_name)
                os.makedirs(ss_dir, exist_ok=True)
                ss_path = os.path.join(
                    ss_dir,
                    f"{ts}_{prefix_tag}_{stage_tag}_{suffix}_frame{frame}.png" if (prefix_tag or stage_tag)
                    else f"{ts}_{suffix}_frame{frame}.png"
                )
            elif not os.path.isabs(ss_path):
                ss_path = os.path.join(PROJECT_ROOT, ss_path)
            os.makedirs(os.path.dirname(ss_path) or PROJECT_ROOT, exist_ok=True)
            fig.savefig(ss_path, dpi=150)
            print(f"\n[INFO] 最终场景截图已保存: {os.path.abspath(ss_path)}")

            # 速度图
            ts2 = datetime.now().strftime("%m%d_%H%M%S")
            suffix = "3d" if MODE_3D else "2d"
            prefix_tag = run_cfg.get('run_prefix', '') or 'eval'
            stage_tag = run_cfg.get('stage', '') or ''
            vel_save_dir = os.path.join(PROJECT_ROOT, "eval_results_apf", scenario_name)
            os.makedirs(vel_save_dir, exist_ok=True)
            vel_name = (
                f"{ts2}_{prefix_tag}_{stage_tag}_{suffix}_velocities.png"
                if (prefix_tag or stage_tag)
                else f"{ts2}_{suffix}_velocities.png"
            )
            vel_path = os.path.join(vel_save_dir, vel_name)
            plot_velocities(velocities_magnitude, velocities_x, velocities_y,
                            headless=ARGS.headless, save_path=vel_path)

            if not ARGS.headless:
                # 给用户一点时间看图
                print("\n(关闭图像窗口即可退出)")
            else:
                ani.event_source.stop()

        return []

    # 8. 动画
    ani = animation.FuncAnimation(
        fig, update,
        frames=max(2, ARGS.max_frames + 10),
        interval=30,    # 30ms/帧 ≈ 33fps
        blit=False,     # 3D 模式不支持 blit
    )

    if not ARGS.headless:
        plt.show()
    else:
        # 无头模式：手动推进动画直到结束
        print("[INFO] 无头模式：正在推进仿真 ...")
        # 画首帧以触发初始化
        fig.canvas.draw()
        # 无头模式下，直接通过循环调用 update，直到 finished=True
        import warnings as _w
        _w.filterwarnings('ignore')
        for f in range(ARGS.max_frames + 5):
            update(f)
            if finished:
                break

    # 确保无头模式下所有 figure 都被清理
    try:
        plt.close(fig)
    except Exception:
        pass

    print(f"\n✓ evaluate_arvp_apf.py 评估结束。")
