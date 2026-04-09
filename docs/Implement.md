# 实现细节 (Implementation Details)

本文档按算法分块说明 ScoReFlow 项目里各类微调方法的关键超参与代码入口。
项目结构上,所有 RL 微调 agent 都继承自 [`agent/finetune/train_agent.py`](../agent/finetune/train_agent.py),
模型实现集中在 [`model/flow/ft_ppo/`](../model/flow/ft_ppo/)。

---

## 1. ScoReFlow (Score-SDE + GammaNet) — 本项目核心方法

ScoReFlow 在 flow matching 策略的 RL 微调中,联合优化 drift(速度场)与 diffusion(噪声),通过解析推导的 score 函数 + 一个轻量可学习的时间调度网络 **GammaNet** 实现完全分布控制。

策略更新遵循:
$$
dx_t = \big[v_\theta(x_t,t) + \alpha_\psi(t)\cdot s(x_t,t)\big]\,dt + \sqrt{2\,\alpha_\psi(t)}\,dW_t
$$

其中 $s(x_t,t)=\dfrac{t\cdot v_\theta-x_t}{1-t}$ 为解析 score, $\alpha_\psi(t)$ 是 GammaNet。

### 关键代码

| 组件 | 路径 |
|---|---|
| 1-ReFlow + Score-SDE + GammaNet | [`model/flow/ft_ppo/ppoflow_with_score_gammanet.py`](../model/flow/ft_ppo/ppoflow_with_score_gammanet.py) |
| ShortCut + Score-SDE + GammaNet | [`model/flow/ft_ppo/pposhortcut_with_score_gammanet.py`](../model/flow/ft_ppo/pposhortcut_with_score_gammanet.py) |
| 通用 score 函数计算 | [`model/flow/score_utils.py`](../model/flow/score_utils.py) |
| 训练 agent (state) | [`agent/finetune/reinflow/train_ppo_flow_agent_score.py`](../agent/finetune/reinflow/train_ppo_flow_agent_score.py) |
| 训练 agent (image) | [`agent/finetune/reinflow/train_ppo_flow_img_agent_score.py`](../agent/finetune/reinflow/train_ppo_flow_img_agent_score.py) |
| ShortCut GammaNet agent | [`agent/finetune/reinflow/train_ppo_shortcut_gammanet_agent.py`](../agent/finetune/reinflow/train_ppo_shortcut_gammanet_agent.py) |

### ScoReFlow 专属超参

- `gamma_score`:score 引导的整体强度 $\alpha_\psi$ 的初始尺度。常用 `1.0`,可在 `[0.1, 2.0]` 之间扫。
- `score_scheduler_type`:GammaNet 时间调度类型(代码里默认 `learn`)。
- `model.gamma_net.hidden_dims`:GammaNet 隐藏层维度,默认轻量 `[32, 32]` 即可。
- `model.gamma_net.activation`:激活函数,默认 `silu`,末端 `softplus` 保证 $\alpha\geq 0$。
- 边界处理:GammaNet 对 $t=1$ 处做 `(1-t)` 硬掩码,防止 $s(x_t,t)$ 在终点的奇异性。

### 与已有方法的关系

| 方法 | 是否有 score | $\alpha$ 是否可学 | 配置示例 |
|---|---|---|---|
| ReinFlow / DPPO | ❌(只改 drift) | — | `ft_ppo_reflow_mlp.yaml` |
| Fixed Score-SDE | ✅ | ❌ 常数 | `ft_ppo_*_with_score_gammanet_const.yaml` |
| ScoReFlow (本项目) | ✅ | ✅ GammaNet | `ft_ppo_*_with_score_gammanet.yaml` |

### 几个常见 ablation 配置入口

- **冻结 v_θ 只学 GammaNet**:[`pposhortcut_with_score_gammanet_frozen_v.py`](../model/flow/ft_ppo/pposhortcut_with_score_gammanet_frozen_v.py)
- **GammaNet 用观测条件**:[`ppoflow_with_score_gammanet_obs.py`](../model/flow/ft_ppo/ppoflow_with_score_gammanet_obs.py)
- **GammaNet 与 v_θ 解耦的双路损失**:[`pposhortcut_with_score_gammanet_dloss.py`](../model/flow/ft_ppo/pposhortcut_with_score_gammanet_dloss.py)

---

## 2. GRPO (Group-Relative Policy Optimization) — Critic-Free 微调

GRPO 是 ScoReFlow 最近新增的 critic-free 替代方案。完全去掉 critic 网络,改用 episode 级 group-relative advantage:
$$
A_i = \dfrac{R_i - \mathrm{mean}(R)}{\mathrm{std}(R)}
$$
并显式加 KL 惩罚替代 value loss。

### 关键代码

| 组件 | 路径 |
|---|---|
| GRPO 经验回放 buffer | [`agent/finetune/reinflow/buffer_grpo.py`](../agent/finetune/reinflow/buffer_grpo.py) |
| GRPO 训练 agent (state) | [`agent/finetune/reinflow/train_grpo_flow_agent.py`](../agent/finetune/reinflow/train_grpo_flow_agent.py) |
| GRPO 训练 agent (image) | [`agent/finetune/reinflow/train_grpo_flow_img_agent.py`](../agent/finetune/reinflow/train_grpo_flow_img_agent.py) |
| GRPO + Score-SDE 模型 | [`model/flow/ft_ppo/grpoflow_with_score_gammanet.py`](../model/flow/ft_ppo/grpoflow_with_score_gammanet.py) |
| GRPO 基线模型 | [`model/flow/ft_ppo/grpoflow.py`](../model/flow/ft_ppo/grpoflow.py) |

### GRPO 专属超参

- `train.kl_coef`:KL 惩罚系数。state 任务推荐 `0.04`,image 任务推荐 `0.1 ~ 0.2`。过大会让策略停止探索,过小会偏离 base policy 过快。
- `train.ent_coef`:熵正则系数。GRPO 下推荐 `0.01`(image)/ `0.03`(state)。
- 与 PPO 的主要差异:**没有** `train.gamma`、`train.gae_lambda`、`train.value_loss_coef`、`model.critic.*`。

### 触发方式

```bash
# Robomimic GRPO + Score-SDE
TASK=square SEED=42 KL_COEF=0.2 \
    bash scripts/train/robomimic/train_robomimic_finetune-grpo.sh

# Kitchen GRPO
TASK=kitchen SEED=42 KL_COEF=0.04 \
    bash scripts/train/gym/train_gym_finetune-grpo.sh
```

---

## 3. ReinFlow / Flow Matching PPO 基线 (drift-only)

drift-only 的 PPO 微调,不使用 score 修正。代码入口:

| 组件 | 路径 |
|---|---|
| FlowMLP / NoisyFlowMLP / VisionFlowMLP | [`model/flow/mlp_flow.py`](../model/flow/mlp_flow.py) |
| ShortCutFlowMLP / NoisyShortCutFlowMLP | [`model/flow/mlp_shortcut.py`](../model/flow/mlp_shortcut.py) |
| state agent | [`agent/finetune/reinflow/train_ppo_flow_agent.py`](../agent/finetune/reinflow/train_ppo_flow_agent.py) |
| image agent | [`agent/finetune/reinflow/train_ppo_flow_img_agent.py`](../agent/finetune/reinflow/train_ppo_flow_img_agent.py) |

### 关键超参

- `min_std` / `max_std`:每个去噪步注入噪声的标准差上下限。
- `denoising_steps`:去噪步数(预训练与微调一致)。
- `ft_denoising_steps`:从最后一步往前数,真正参与微调的步数(默认 = `denoising_steps`)。
- `train.clip_intermediate_actions`:微调与评估都建议开启,防止中间动作越界。
- `model.denoised_clip_value`:去噪动作的最大绝对值,默认 `1`。
- `model.randn_clip_value`:每步采样的标准差倍数上限,默认 `3`。
- `model.clip_ploss_coef`:PPO 裁剪 $\epsilon$。state 任务用 `0.01`,visual 任务用 `0.001`。
- `model.logprob_min` / `model.logprob_max`:对数概率裁剪范围。策略坍塌时 logprob 会变得极小(很负),此时应**降低**噪声水平。
- `model.noise_scheduler_type`:噪声调度,推荐 state locomotion 用 `learn_decay`,固定噪声用 `constant`。
- `model.use_time_independent_noise`:噪声是否依赖观测和时间。
- `model.critic.out_bias_init`:critic 初始化偏差。如果 critic 初始输出为负但策略已有 30% 成功率,设置为正值(我们在 transport image 任务设为 4.0)。
- `train.use_bc_loss` / `train.bc_loss_type` / `train.bc_loss_coeff`:BC 正则,通常只在 hopper 上需要。
- `train.ent_coef`:熵系数。state 任务用 `0.03`,image 任务用 `0`。

---

## 4. DPPO 基线 (Diffusion Policy + PPO)

| 组件 | 路径 |
|---|---|
| DPPO 训练 agent | [`agent/finetune/dppo/train_ppo_diffusion_agent.py`](../agent/finetune/dppo/train_ppo_diffusion_agent.py) |
| ReinFlow 改写版(支持 resume / 更细日志) | [`agent/finetune/reinflow/train_ppo_diffusion_agent.py`](../agent/finetune/reinflow/train_ppo_diffusion_agent.py) |
| 图像版 | [`agent/finetune/reinflow/train_ppo_diffusion_img_agent.py`](../agent/finetune/reinflow/train_ppo_diffusion_img_agent.py) |

### 关键超参

- `denoising_steps`:去噪步数(预训练 / 微调一致)。
- `ft_denoising_steps`:实际参与微调的步数。
- `horizon_steps`:动作 chunk 大小(MLP 应等于 `act_steps`)。
- `model.gamma_denoising`:去噪步折扣因子。
- `model.min_sampling_denoising_std`、`model.min_logprob_denoising_std`。
- `model.clip_ploss_coef`:PPO 裁剪。
- `train.batch_size`:DPPO 的 batch 较大(对环境步与去噪步同时取期望)。

### DDIM 微调

预训练 `denoising_steps=100`,微调时:`model.use_ddim=True`,`model.ddim_steps=<目标步数>`,`ft_denoising_steps=<目标微调步数>`。

---

## 5. FQL (Offline-to-Online Flow Q-Learning) 基线

| 超参 | 含义 |
|---|---|
| `offline_steps` | 离线微调迭代数 |
| `online_steps` | 在线微调迭代数 |
| `eval_base_model` | 调试用,周期评估 base policy 演化情况 |

入口示例:`bash scripts/train/robomimic/train_square_fql.sh`

---

## 6. Offline RL 基线 (Cal-QL / IBRL / RLPD)

继承自 DPPO,代码位于 [`agent/finetune/offlinerl_baselines/`](../agent/finetune/offlinerl_baselines/):

- [`train_calql_agent.py`](../agent/finetune/offlinerl_baselines/train_calql_agent.py)
- [`train_ibrl_agent.py`](../agent/finetune/offlinerl_baselines/train_ibrl_agent.py)
- [`train_rlpd_agent.py`](../agent/finetune/offlinerl_baselines/train_rlpd_agent.py)

---

## 7. Diffusion x RL 基线 (RWR / DAWR / DIPO / DQL / IDQL / QSM)

继承自 DPPO,代码位于 [`agent/finetune/diffusion_baselines/`](../agent/finetune/diffusion_baselines/),与 ScoReFlow 公平对比时使用。

---

## 8. 想自己加一个新的 RL 算法?

继承 [`agent/finetune/train_agent.py`](../agent/finetune/train_agent.py) 的 `TrainAgent` 基类,
然后在 `cfg/.../ft_<your_algo>_*.yaml` 写一个 Hydra 配置,
通过 `_target_:` 字段指向你的类即可。`run.py` 会自动用 `hydra.utils.get_class(cfg._target_)` 反射加载。
