## 自定义数据集 / 环境

### 预训练数据

预训练入口为 [`agent/pretrain/train_diffusion_agent.py`](../agent/pretrain/train_diffusion_agent.py),数据加载器在 [`agent/dataset/sequence.py`](../agent/dataset/sequence.py)。数据集需要是 `.npz` 文件,包含以下 numpy 数组:

| 字段 | 形状 | 说明 |
|---|---|---|
| `states` | `num_total_steps × obs_dim` | 状态序列 |
| `actions` | `num_total_steps × act_dim` | 动作序列 |
| `images` | `num_total_steps × C × H × W` | 图像观测(可选,多相机时通道维拼接;`H == W` 且为 8 的倍数) |
| `traj_lengths` | `1-D` | 用于切分轨迹边界的索引 |

#### 用现成的 D4RL 数据(Gym / Kitchen)

从 [Hugging Face 镜像](https://huggingface.co/datasets/imone/D4RL/tree/main) 下载 raw `.hdf5`,然后用本仓库的工具脚本转 `.npz`:

```bash
# 1. 查看 hdf5 内部结构
python data_process/read_hdf5.py --file_path=<PATH_TO_YOUR_HDF5>

# 2. 转 npz + 归一化到 [-1, 1],输出 train.npz / normalization.npz
python data_process/hdf5_to_npz.py --data_path=<PATH_TO_YOUR_HDF5>

# 3. 检查转换结果
python data_process/read_npz.py --data_path=<DIR>/train.npz
```

> 也可以直接在预训练命令里加 `use_d4rl_dataset=True`,会自动下载并处理。

#### Robomimic 图像数据

在预训练命令里 `cfg/robomimic/pretrain/<task>/pre_*.yaml` 已配置好下载链接,首次运行 `python run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_shortcut_mlp_img` 会自动从 Google Drive 拉取数据集与归一化统计。

### 历史观测

ScoReFlow 默认只使用当前时刻的观测,但代码已支持历史观测拼接。在配置里设置:

```yaml
cond_steps: 4         # 状态历史步数
img_cond_steps: 2     # 图像历史步数(必须 ≤ cond_steps)
```

预训练和微调时保持一致即可。

### 微调环境

环境遵循 Gym 接口。向量化环境初始化入口在 [`env/gym_utils/__init__.py:make_async`](../env/gym_utils/__init__.py),由父类 [`agent/finetune/train_agent.py`](../agent/finetune/train_agent.py) 调用。

ScoReFlow 实际使用的 wrapper:
- [`env/gym_utils/wrapper/multi_step.py`](../env/gym_utils/wrapper/multi_step.py) — 历史观测堆叠 + 多步动作执行
- [`env/gym_utils/wrapper/robomimic_lowdim.py`](../env/gym_utils/wrapper/robomimic_lowdim.py) — Robomimic 状态归一化
- [`env/gym_utils/wrapper/mujoco_locomotion_lowdim.py`](../env/gym_utils/wrapper/mujoco_locomotion_lowdim.py) — D4RL Gym/Kitchen 归一化

要接入新环境,实现一个新的 wrapper 并在 `make_async` 中加分支即可。

