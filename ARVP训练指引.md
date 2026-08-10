# ARVP-MADDPG 训练指引

> 本指引基于论文《RVO-Guided RL Framework with Multi-Head Attention for Multi-UAV Pursuit and Dynamic Obstacle Avoidance》与仓库源码（`/workspace`）整理，手把手教你完成 **ARVP 算法第一阶段、第二阶段**训练，以及 **对比算法 / 消融模型** 的训练。

---

## 0. 快速结论（先看这里）

- **两阶段课程学习的关键开关**只有一个：环境奖励函数里 `mu3`（三阶段围捕奖励 `r_stage` 的系数）。
  - 第一阶段（动态避障）：`mu3 = 0.0`
  - 第二阶段（围捕任务）：`mu3 = 0.4`
- 位置：[env/sim_env_rvo.py](file:///workspace/env/sim_env_rvo.py) 的 `cal_rewards_dones` 方法（约第 976 行）。
- 第一阶段从零训练（注释掉 `load_checkpoint`），第二阶段加载第一阶段权重继续训练。
- ARVP 主训练脚本：[train_arvp.py](file:///workspace/train_arvp.py)。

---

## 1. 环境与依赖准备

### 1.1 硬件 / 软件
- GPU：论文使用 NVIDIA RTX 2070 SUPER；无 GPU 也可跑（CPU 会慢）。
- Python 3.9、PyTorch 2.0.0、CUDA 11.8。

### 1.2 安装依赖
```bash
pip install torch gym numpy pandas matplotlib tqdm pillow tensorboard sympy pypdf
```

### 1.3 任务场景（论文设定）
- 封闭 2D 环境，边界 2m × 2m。
- 4 个智能体：3 架四旋翼无人机（追捕者）+ 1 个红点目标（逃跑者）。
- 4 个圆形动态障碍物，匀速直线运动、碰壁反射。
- 每回合最多 110 步（`MAX_STEPS`），时间步长 Δt = 0.5s。

---

## 2. 代码结构地图（算法 ↔ 文件对应）

理解这张表，就理解了整个仓库：

| 算法 | 训练脚本 | 模型 | 环境 | 经验回放 | 检查点目录 |
|---|---|---|---|---|---|
| **ARVP-MADDPG**（本文方法） | [train_arvp.py](file:///workspace/train_arvp.py) | [model/maddpg_att.py](file:///workspace/model/maddpg_att.py)（含多头注意力） | [env/sim_env_rvo.py](file:///workspace/env/sim_env_rvo.py)（含 RVO 奖励） | [buffer/PERbuffer.py](file:///workspace/buffer/PERbuffer.py)（优先回放） | `tmp_avoid_dynamic/maddpgwithatt/` |
| **MADDPG**（基线对比） | [train_base.py](file:///workspace/train_base.py) | [model/maddpg.py](file:///workspace/model/maddpg.py) | [env/sim_env.py](file:///workspace/env/sim_env.py)（无 RVO 奖励） | [buffer/buffer.py](file:///workspace/buffer/buffer.py)（均匀回放） | `tmp/maddpg/` |
| **MADDPG-LSTM-PRE**（LSTM 变体，无注意力） | [train_lstmrnn.py](file:///workspace/train_lstmrnn.py) | [model/maddpg_lstm_pre.py](file:///workspace/model/maddpg_lstm_pre.py) | env/sim_env_rvo.py | [buffer/RNNPREbuffer.py](file:///workspace/buffer/RNNPREbuffer.py) | `tmp_test/maddpgwithatt/` |
| **MADDPG-ATT-LSTM-PRE**（LSTM+注意力 变体） | [train_lstmrnn_att.py](file:///workspace/train_lstmrnn_att.py) | [model/maddpg_att_lstm_pre.py](file:///workspace/model/maddpg_att_lstm_pre.py) | env/sim_env_rvo.py | buffer/RNNPREbuffer.py | `tmp_test/maddpgwithatt/` |
| **RVO**（纯几何法，非 RL） | 无训练，直接评测 | — | — | — | — |
| **MATD3**（论文对比项） | ⚠️ 仓库未提供实现 | — | — | — | — |

> **组件含义**：ATT=多头注意力（[ATT.py](file:///workspace/ATT.py)，`head_count=4`）；PER=优先经验回放（`α=0.6, β=0.4, ε=1e-5`）；RVO=交互速度障碍奖励（环境里的 `VO_reward`，系数 `mu5=0.1`）。

---

## 3. 两阶段课程学习原理

论文将训练分为两阶段，由环境奖励系数 `mu3`（`r_stage` 三阶段围捕奖励）控制：

| 阶段 | 目标 | `mu3` 取值 | 其它系数 | 训练回合 |
|---|---|---|---|---|
| **第一阶段** | 学会动态避障 + 接近目标 | `0.0`（关闭围捕奖励） | `mu1=0.9, mu2=0.2, mu5=0.1, mu4=10` | ≈5000 |
| **第二阶段** | 在避障基础上学会包围 + 捕获 | `0.4`（开启围捕奖励） | 同上 | ≈5000 |

**代码位置**：[env/sim_env_rvo.py](file:///workspace/env/sim_env_rvo.py) 的 `cal_rewards_dones` 方法：

```python
mu1 = 0.9 # r_near 0.9
mu2 = 0.2 # r_safe  0.4
mu3 = 0.4 # r_multi_stage 0.0   ← 这就是阶段开关！0.0=阶段1，0.4=阶段2
mu4 = 10  # r_finish 10
mu5 = 0.1 # 避障 0.2            ← RVO 奖励系数，两阶段都保持 0.1
```

> 注：论文表 1 的系数记号（µ1~µ5）与代码变量下标并非一一对应，但**代码是训练的实际依据**。代码中 `mu3` 即围捕奖励 `r_stage` 的系数，注释里的 `0.0` 即第一阶段的取值。

---

## 4. ARVP 第一阶段训练（动态避障）

### 步骤 1：切换奖励为第一阶段配置
编辑 [env/sim_env_rvo.py](file:///workspace/env/sim_env_rvo.py)，把 `mu3` 改为 `0.0`：

```python
mu3 = 0.0 # r_multi_stage   阶段1：关闭围捕奖励
```

### 步骤 2：让训练从零开始（不加载旧权重）
编辑 [train_arvp.py](file:///workspace/train_arvp.py)，第 96 行处：

```python
print(chkpt_dir)
# maddpg_agents.load_checkpoint()   ← 注释掉这一行（第一阶段从零训练）
```

> 若 `tmp_avoid_dynamic/maddpgwithatt/` 下已有旧权重想重新开始，先备份/删除该目录。

### 步骤 3：确认超参数（[train_arvp.py](file:///workspace/train_arvp.py)）
```python
alpha = 0.00001      # actor 学习率 l_α
beta  = 0.001         # critic 学习率 l_β
N_GAMES = 6000        # 总回合数（论文约 5000，可按需调）
MAX_STEPS = 110
BATCH_SIZE = 100      # 每 100 回合统计一次并保存
```
> 论文第一阶段设定 5000 回合。如想严格对齐，把 `N_GAMES` 设为 5000。

### 步骤 4：启动训练
```bash
cd /workspace
python train_arvp.py
```
- 训练日志写入 `runs/UAV_trainingXX`（TensorBoard）。
- 每 100 回合打印平均奖励；当平均奖励创新高时自动 `save_checkpoint()` 到 `tmp_avoid_dynamic/maddpgwithatt/UAV_Round_up/`。
- 观察训练曲线：
```bash
tensorboard --logdir runs
```

### 步骤 5：第一阶段验收
第一阶段**不追求完成率**（论文表 3 显示完成率为 0），重点关注：
- **碰撞率**应降到很低（论文最优 3.74%）。
- **稳态误差**（到目标距离）应较小（论文 0.46m）。
- 奖励曲线趋于收敛。

确认 `tmp_avoid_dynamic/maddpgwithatt/UAV_Round_up/` 下生成了 `agent_X_actor / agent_X_critic / target_*` 等权重文件，即可进入第二阶段。

---

## 5. ARVP 第二阶段训练（围捕任务）

### 步骤 1：切换奖励为第二阶段配置
编辑 [env/sim_env_rvo.py](file:///workspace/env/sim_env_rvo.py)，把 `mu3` 改回 `0.4`：

```python
mu3 = 0.4 # r_multi_stage   阶段2：开启围捕奖励
```

### 步骤 2：加载第一阶段权重
编辑 [train_arvp.py](file:///workspace/train_arvp.py)，第 96 行处恢复加载：

```python
print(chkpt_dir)
maddpg_agents.load_checkpoint()   ← 取消注释，加载第一阶段权重
```
> 确保第一阶段权重仍在 `tmp_avoid_dynamic/maddpgwithatt/UAV_Round_up/`。

### 步骤 3：启动训练
```bash
cd /workspace
python train_arvp.py
```
- 第二阶段引入 `r_track / r_encircle / r_capture`，模型在保持避障的同时学习包围与捕获。
- 训练中每当平均奖励创新高，会自动调用 `evaluate_arvp.py` 跑一次快速评测。

### 步骤 4：第二阶段验收（论文表 2 / 表 4 目标）
| 指标 | 论文 ARVP-MADDPG |
|---|---|
| 完成率 | ≈96% |
| 碰撞率 | ≈7.28% |
| 稳态误差 | ≈0.085–0.096m |

---

## 6. 评测 / 可视化

训练完成后，加载最终权重跑完整回合并输出指标：

```bash
cd /workspace
python evaluate_arvp.py
```
- [evaluate_arvp.py](file:///workspace/evaluate_arvp.py) 加载 `tmp_avoid_dynamic/maddpgwithatt/` 权重，用 `evaluate=True` 选择动作（无探索噪声）。
- 动画展示无人机轨迹，结束后打印每架 UAV 的评测结果（路径长度、平滑度、碰撞等）。
- 评测指标定义见论文 §VI-B（完成率/碰撞率/路径长度/平滑度/稳态误差）。

> `evaluate = True` 时会每 10 步保存截图到 `images/episode_*_step_*.png`。

---

## 7. 对比算法训练

### 7.1 MADDPG（基线）
```bash
cd /workspace
python train_base.py
```
- 用 [train_base.py](file:///workspace/train_base.py)：无注意力、无 PER、无 RVO 奖励环境（[env/sim_env.py](file:///workspace/env/sim_env.py)，其 `mu3=0`、无 `VO_reward`）。
- 权重存 `tmp/maddpg/`。评测：`python evaluate_base.py`。
- 超参：`alpha=0.00001, beta=0.02, N_GAMES=5000`。
- 注：`train_base.py` 的 `env.step` 只返回 3 元组，与 ARVP 的 5 元组不同，已适配各自脚本。

### 7.2 MATD3（论文对比项）
⚠️ **本仓库未提供 MATD3 实现**。若需复现论文对比，可自行实现：
- 在 MADDPG 基础上，把 Critic 改为**双 Q 网络（Twin Critics）**，取两者最小值作为目标：
  `y = r + γ · min(Q1', Q2')`。
- Actor 用 TD3 风格的延迟更新与目标策略平滑。
- 可基于 [model/maddpg.py](file:///workspace/model/maddpg.py) 与 [model/networks.py](file:///workspace/model/networks.py) 改造，复用 [train_base.py](file:///workspace/train_base.py) 流程。

### 7.3 RVO（纯几何法）
RVO 不是强化学习，无需训练，直接评测避障效果。仓库提供 [RVO.py](file:///workspace/RVO.py)（RVO 速度障碍计算）与 [env/sim_env_rvo.py](file:///workspace/env/sim_env_rvo.py) 中的 `VO_reward`/`VO_Plot`。论文图 3 即对比“RVO 算法轨迹”与“RVO 奖励引导（本文）轨迹”。可编写一个用 RVO 速度作为动作的评测脚本来复现。

### 7.4 LSTM 变体（进阶对比）
```bash
python train_lstmrnn_att.py     # MADDPG-ATT-LSTM-PRE（含注意力）
python train_lstmrnn.py         # MADDPG-LSTM-PRE（无注意力）
```
- 用 [train_lstmrnn_att.py](file:///workspace/train_lstmrnn_att.py) / [train_lstmrnn.py](file:///workspace/train_lstmrnn.py)。
- Actor 带 4 层 LSTM（`lstm_num_layers=4, lstm_hidden_dim=256`），用 `RNNPERMultiAgentReplayBuffer`。
- 权重存 `tmp_test/maddpgwithatt/`，评测 `python evaluate_rnn.py`。
- `N_GAMES=50000`（LSTM 收敛慢，回合数更大）。

---

## 8. 消融实验（论文表 3 / 表 4）

消融通过**组合替换**三个组件实现，无需写新脚本，改导入即可。在 [train_arvp.py](file:///workspace/train_arvp.py) 顶部按需替换：

| 消融模型 | 含义 | model 导入 | buffer 导入 | env 导入 |
|---|---|---|---|---|
| **ARVP-MADDPG**（完整） | ATT+PER+RVO | `model.maddpg_att` | `buffer.PERbuffer` | `env.sim_env_rvo` |
| **MADDPG-PER-RVO** | 去注意力 | `model.maddpg` | `buffer.PERbuffer` | `env.sim_env_rvo` |
| **MADDPG-ATT-RVO** | 去 PER | `model.maddpg_att` | `buffer.buffer`(均匀) | `env.sim_env_rvo` |
| **MADDPG-ATT-PER** | 去 RVO | `model.maddpg_att` | `buffer.PERbuffer` | `env.sim_env`(无 RVO) |

替换示例（以“去 PER”为例）：
```python
# from buffer.PERbuffer import PERMultiAgentReplayBuffer
from buffer.buffer import MultiAgentReplayBuffer
...
memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims,
                    n_actions, n_agents, batch_size=256)
```
> 每个消融模型也需走两阶段（`mu3` 同 ARVP），建议各自用独立 `chkpt_dir` 避免权重覆盖。

---

## 9. 超参数速查表（对齐论文表 1）

| 参数 | 值 | 代码位置 |
|---|---|---|
| 演员学习率 `alpha` (l_α) | 1e-5 | train_arvp.py |
| 评论家学习率 `beta` (l_β) | 1e-3 | train_arvp.py |
| 折扣因子 `gamma` | 0.99 | model/maddpg_att.py 默认 |
| 软更新率 `tau` | 0.005 | model 默认 |
| 回放池容量 `M_b` | 1,000,000 | train_arvp.py |
| 采样批大小 `N_b` | 256 | train_arvp.py / buffer |
| PER `alpha` (优先级敏感度) | 0.6 | buffer/PERbuffer.py |
| PER `beta` (重要性采样) | 0.4 | buffer/PERbuffer.py |
| PER `epsilon` | 1e-5 | buffer/PERbuffer.py |
| 注意力头数 `K_h` | 4 | ATT.py (`head_count=4`) |
| 捕获距离阈值 `d_capture` | 0.3 | env/sim_env_rvo.py |
| 包围距离阈值 `d_limit` | 0.55 | env/sim_env_rvo.py |
| 无人机最大速度 `v_m` | 0.12 | env |
| 每回合步数 `L_ep` | 110 | train_arvp.py `MAX_STEPS` |
| 奖励系数 `mu1..mu5` | 0.9 / 0.2 / 0.4(阶段2) / 10 / 0.1 | env/sim_env_rvo.py |

---

## 10. 常见问题

**Q1：`load_checkpoint()` 报错说找不到文件？**
A：第一阶段从零训练时该目录无权重。注释掉 `load_checkpoint()` 即可；或先跑第一阶段生成权重。

**Q2：第二阶段奖励不升反降？**
A：检查 `mu3` 是否已改回 `0.4`；确认加载的是第一阶段权重（打印 `chkpt_dir` 路径核对）。

**Q3：CPU 太慢？**
A：代码已自动检测 GPU（`torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')`）。确保装了 CUDA 版 PyTorch。LSTM 变体尤其吃算力。

**Q4：`train_base.py` 报 `env.step` 返回值数量不对？**
A：`train_base.py` 与 [env/sim_env.py](file:///workspace/env/sim_env.py) 配套（返回 3 元组），不要混用 `sim_env_rvo`。ARVP 用 `sim_env_rvo`（返回 5 元组）。

**Q5：评测时无人机不动 / 乱飞？**
A：确认 `evaluate=True` 且加载的是第二阶段权重；`evaluate_arvp.py` 里 `chkpt_dir` 必须与训练时一致（`tmp_avoid_dynamic/maddpgwithatt/`）。

---

## 11. 推荐训练流程一览

```
第一阶段（避障）
  env/sim_env_rvo.py: mu3 = 0.0
  train_arvp.py: 注释 load_checkpoint()
  python train_arvp.py  (≈5000 回合)
       ↓ 保存权重到 tmp_avoid_dynamic/

第二阶段（围捕）
  env/sim_env_rvo.py: mu3 = 0.4
  train_arvp.py: 恢复 load_checkpoint()
  python train_arvp.py  (≈5000 回合)
       ↓

评测
  python evaluate_arvp.py
       ↓
对比算法: python train_base.py  (MADDPG)
          自行实现 MATD3
          RVO 直接评测
```

完成上述流程即可复现论文中 ARVP-MADDPG 及其对比/消融实验的结果。
