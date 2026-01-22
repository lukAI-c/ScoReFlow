#!/usr/bin/env python3
"""
带放大窗口的绘图脚本 - 复用 plot_universal.py 的逻辑
用法:
python agent/eval/visualize/plot_with_zoom.py     environment_name=gym-state     task_name=humanoid-d4rl     plot_x_axis=wallclock     +zoom_xlim=[0,1.7]     +zoom_ylim=[4000,5250]     +zoom_position=lower_right +zoom_pos=[0.35,0.05] +zoom_size=[0.6,0.6]
"""

import os
import sys
import re
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle, ConnectionPatch

# 添加项目路径
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, _PROJECT_ROOT)

from agent.eval.visualize.constants import method_name_dict, color_dict, time_step_ratios

# 常量
PPO_EVAL_FREQ = 10
FQL_EVAL_FREQ = 5000
FQL_LOG_FREQ = 200


def load_data(environment_name, task_name, csv_filename, re_expression):
    """加载并处理数据"""
    csv_path = os.path.join(_PROJECT_ROOT, f'visualize/Final_experiments/data/finetune/{environment_name}/{task_name}/{csv_filename}')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    data = pd.read_csv(csv_path)
    if 'kitchen' in task_name:
        data.loc[:, data.columns != 'Step'] *= 0.25

    method_config = method_name_dict[task_name]

    # 提取方法
    method_seed_map = {}
    for col in data.columns:
        if ('success rate' in col.lower() or 'episode reward' in col.lower()) and '__MIN' not in col and '__MAX' not in col:
            match = re.search(re_expression, col)
            if match:
                method = match.group(1)
                seed = int(match.group(2) or match.group(3) or match.group(4) or 0)
                method_seed_map.setdefault(method, []).append((seed, col))

    # 映射显示名称
    updated_map = {}
    for old, seed_cols in method_seed_map.items():
        name = old
        for cfg in method_config:
            if cfg['original_name'] == old:
                name = cfg['display_name']
                break
        updated_map[name] = seed_cols

    # 计算统计 - 同时保存每个方法对应的有效 Step 索引
    method_stats = {}
    for method, seed_cols in updated_map.items():
        # 找到所有 seed 都有有效数据的行索引
        valid_indices_list = []
        for _, col in seed_cols:
            valid_idx = data[col].dropna().index.tolist()
            valid_indices_list.append(set(valid_idx))

        if not valid_indices_list:
            continue

        # 取所有 seed 的交集（共同有效的行）
        common_indices = sorted(list(set.intersection(*valid_indices_list)))
        if not common_indices:
            continue

        # 提取数据
        rates = []
        for _, col in seed_cols:
            rate = data.loc[common_indices, col].values
            rates.append(rate)

        arr = np.array(rates)
        # 保存统计数据和对应的 Step 值
        method_stats[method] = {
            'mean': np.nanmean(arr, axis=0),
            'std': np.nanstd(arr, axis=0),
            'steps': data.loc[common_indices, 'Step'].values  # 保存对应的 Step
        }

    # 颜色映射
    color_map = {cfg['display_name']: cfg['color'] for cfg in method_config}
    for m in updated_map:
        if m not in color_map:
            color_map[m] = color_dict.get(m, 'gray')

    return data, method_stats, color_map


def create_zoom_inset(ax, zoom_xlim, zoom_ylim, zoom_position='lower right', zoom_size=(0.45, 0.45), zoom_pos=None):
    """
    创建放大窗口

    Args:
        ax: 主图 axes
        zoom_xlim: 放大区域 x 范围
        zoom_ylim: 放大区域 y 范围
        zoom_position: 预设位置 ('upper left', 'upper right', 'lower left', 'lower right')
        zoom_size: 子图大小 [width, height]，范围 0-1
        zoom_pos: 自定义位置 [x, y]，范围 0-1，左下角为原点。如果指定则覆盖 zoom_position
    """
    # 预设位置映射
    pos_map = {
        'upper left': [0.12, 0.52], 'upper right': [0.52, 0.52],
        'lower left': [0.12, 0.12], 'lower right': [0.52, 0.12],
    }

    # 使用自定义位置或预设位置
    if zoom_pos is not None:
        pos = zoom_pos
    else:
        pos = pos_map.get(zoom_position, [0.52, 0.12])

    # 创建子图
    axins = ax.inset_axes([pos[0], pos[1], zoom_size[0], zoom_size[1]])
    axins.set_xlim(zoom_xlim)
    axins.set_ylim(zoom_ylim)

    # 设置子图样式：完全移除刻度标签
    axins.grid(True, alpha=0.3)
    axins.set_xticklabels([])  # 移除 x 轴刻度标签
    axins.set_yticklabels([])  # 移除 y 轴刻度标签
    axins.tick_params(axis='both', which='both', length=0)  # 移除刻度线

    # 加粗边框
    for spine in axins.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2.5)

    # 在主图上标记放大区域（红色虚线框）
    rect = Rectangle((zoom_xlim[0], zoom_ylim[0]), zoom_xlim[1]-zoom_xlim[0], zoom_ylim[1]-zoom_ylim[0],
                      fill=False, edgecolor='red', linewidth=2.5, linestyle='--', zorder=10)
    ax.add_patch(rect)

    # 连接线：从放大区域连接到子图
    if 'lower' in zoom_position or (zoom_pos and zoom_pos[1] < 0.4):
        corner = (zoom_xlim[1], zoom_ylim[1])  # 放大区域右上角
        inset_corner = (1, 0)  # 子图上角
    else:
        corner = (zoom_xlim[1], zoom_ylim[0])  # 放大区域右下角
        inset_corner = (0, 1)  # 子图左下角

    con = ConnectionPatch(xyA=corner, coordsA=ax.transData, xyB=inset_corner, coordsB=axins.transAxes,
                          arrowstyle="-", linewidth=2, color='red', linestyle='--', alpha=0.8)
    ax.add_artist(con)
    return axins


@hydra.main(config_path="visualize_cfgs", config_name="final_experiments", version_base="1.1")
def main(cfg: DictConfig):
    """主函数 - 绘制带放大窗口的图"""
    os.chdir(_PROJECT_ROOT)

    # 获取参数
    environment_name = cfg.get('environment_name', 'kitchen')
    task_name = cfg.get('task_name', 'kitchen-complete-v0')
    plot_x_axis = cfg.get('plot_x_axis', 'sample')
    is_wallclock = (plot_x_axis == 'wallclock')

    # 放大区域参数
    zoom_xlim = list(cfg.get('zoom_xlim', [1500000, 2000000]))
    zoom_ylim = list(cfg.get('zoom_ylim', [0.85, 1.0]))
    zoom_position = cfg.get('zoom_position', 'lower_right').replace('_', ' ')
    zoom_size = list(cfg.get('zoom_size', [0.45, 0.45]))  # 默认放大到 0.45
    zoom_pos = list(cfg.get('zoom_pos')) if cfg.get('zoom_pos') else None  # 自定义位置 [x, y]
    custom_xlim = list(cfg.get('custom_xlim')) if cfg.get('custom_xlim') else None
    custom_ylim = list(cfg.get('custom_ylim')) if cfg.get('custom_ylim') else None

    print("=" * 60)
    print(f"Plot with Zoom: {environment_name}/{task_name}")
    print(f"X-axis: {plot_x_axis}")
    print(f"Zoom region: x={zoom_xlim}, y={zoom_ylim}")
    print(f"Zoom size: {zoom_size}")
    if zoom_pos:
        print(f"Zoom position (custom): {zoom_pos}")
    else:
        print(f"Zoom position (preset): {zoom_position}")
    print("=" * 60)

    # 加载配置
    env_cfg = cfg.env[environment_name][task_name]
    csv_filename = env_cfg.get('csv_filename', 'all_methods_merged.csv')
    re_expression_raw = env_cfg.get('re_expression', '')
    n_parallel_envs = env_cfg.get('n_parallel_envs', 1)
    n_rollout_steps = env_cfg.get('n_rollout_steps', 1)
    n_act_steps = env_cfg.get('n_act_steps', 1)

    # 处理正则表达式
    if re_expression_raw.startswith("r''"):
        re_expression = re_expression_raw[3:-2]
    elif re_expression_raw.startswith("r'"):
        re_expression = re_expression_raw[2:-1]
    else:
        re_expression = re_expression_raw

    # 加载数据
    data, method_stats, color_map = load_data(environment_name, task_name, csv_filename, re_expression)
    print(f"Loaded methods: {list(method_stats.keys())}")
    for method, stats in method_stats.items():
        print(f"  {method}: {len(stats['mean'])} points, steps range [{stats['steps'][0]}, {stats['steps'][-1]}]")

    # 设置 x 轴标签
    if plot_x_axis == 'step':
        x_label = 'Steps'
    elif plot_x_axis == 'sample':
        x_label = 'Samples'
    elif plot_x_axis == 'wallclock':
        x_label = 'Wall-Clock Time (hours)'
    else:
        raise ValueError(f"Unsupported plot_x_axis: {plot_x_axis}")

    evaluation_name = 'TaskCompletionRate' if 'kitchen' in task_name else 'AverageEpisodeReward'

    # 创建图形
    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    handles, plot_labels = [], []

    # 绘制主图和放大图的函数
    def plot_curves(target_ax, for_zoom=False):
        for method, stats in method_stats.items():
            mean = stats['mean'].copy()
            std = stats['std'].copy()
            method_steps = stats['steps']  # 每个方法自己的 Step 值
            color = color_map.get(method, 'gray')

            is_ours = '(ours)' in method
            lw = 3 if is_ours else 1.5
            alpha = 1.0 if is_ours else 0.6

            # 计算该方法的 x 轴
            if is_wallclock:
                if method not in time_step_ratios.get(task_name, {}):
                    continue
                time_ratio = time_step_ratios[task_name][method]
                if method == 'FQL':
                    mean, std = mean[::5], std[::5]
                    method_steps = method_steps[::5]
                    eval_interval = FQL_EVAL_FREQ
                    time_per_itr = time_ratio / FQL_LOG_FREQ
                else:
                    eval_interval = PPO_EVAL_FREQ
                    time_per_itr = time_ratio
                method_x = np.arange(len(mean)) * eval_interval * time_per_itr / 3600
            elif plot_x_axis == 'step':
                method_x = method_steps
            elif plot_x_axis == 'sample':
                method_x = method_steps * n_parallel_envs * n_rollout_steps * n_act_steps
            else:
                method_x = method_steps

            line, = target_ax.plot(method_x, mean, linewidth=lw, color=color, alpha=alpha, label=method)
            target_ax.fill_between(method_x, mean - std, mean + std, alpha=0.1, color=color)

            if not for_zoom:
                handles.append(line)
                plot_labels.append(method)

    # 绘制主图
    plot_curves(ax)

    # 设置主图
    ax.set_xlabel(x_label, fontsize=24)
    ax.set_ylabel(evaluation_name, fontsize=24)
    ax.tick_params(axis='both', labelsize=20)
    ax.grid(True)
    if custom_xlim: ax.set_xlim(custom_xlim)
    if custom_ylim: ax.set_ylim(custom_ylim)

    # 创建放大窗口
    axins = create_zoom_inset(ax, zoom_xlim, zoom_ylim, zoom_position, zoom_size, zoom_pos)
    plot_curves(axins, for_zoom=True)

    # 保存
    output_dir = os.path.join(_PROJECT_ROOT, 'visualize/Final_experiments/outs')
    fig_dir = os.path.join(output_dir, environment_name, task_name)
    os.makedirs(fig_dir, exist_ok=True)

    suffix = "_wallclock" if is_wallclock else ""
    output_filename = f"{environment_name}_{task_name}_{evaluation_name}{suffix}_zoom"
    output_path = os.path.join(fig_dir, output_filename)

    fig.savefig(f'{output_path}.png', bbox_inches='tight', dpi=300)
    fig.savefig(f'{output_path}.pdf', bbox_inches='tight')
    print(f"Saved: {output_path}.png / .pdf")
    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    main()

