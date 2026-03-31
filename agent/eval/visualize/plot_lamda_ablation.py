#!/usr/bin/env python3
"""
λ 消融实验绘图脚本

比较三种 λ(t) 控制方式:
  1. Score-SDE (hyperparameter-coupled)          - ppo_shortcut_mlp_score      (固定调度器)
  2. ablation (learned-coupled) - ppo_shortcut_mlp_score_epsnet (MLP 学习，drift/diffusion 耦合)
  3. ours (learned-decoupled) - ppo_shortcut_mlp_with_score_gammanet (GammaNet，drift/diffusion 解耦)

用法:
    python agent/eval/visualize/plot_lamda_ablation.py
    python agent/eval/visualize/plot_lamda_ablation.py --csv visualize/Final_experiments/data/kitchen-complete/compare_lamda.csv
    python agent/eval/visualize/plot_lamda_ablation.py --x_axis sample --ylim 0 4
"""

import os
import sys
import re
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ── 项目根目录 ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, _PROJECT_ROOT)

# ── 消融实验方法配置 ──────────────────────────────────────────────────────────
# original_name: wandb run 名中 "kitchen-complete-v0_" 之后、"_seed***" 之前的部分
METHOD_CONFIG = [
    {
        'original_name': 'ppo_shortcut_mlp_score_ta4_td4',
        'display_name':  r'Score-SDE (hyperparameter-coupled)',
        'color':         '#808080',   # 灰色
        'is_ours':       False,
    },
    {
        'original_name': 'ppo_shortcut_mlp_score_epsnet_ta4_td4_gamma1.0',
        'display_name':  r'ablation (learned-coupled)',
        'color':         '#E8541A',   # 橙红色
        'is_ours':       True,
    },
    {
        'original_name': 'ppo_shortcut_mlp_with_score_gammanet_ta4_td4_tdf4',
        'display_name':  r'ours (learned-decoupled)',
        'color':         '#4335FF',   # 蓝紫色
        'is_ours':       False,
    },
]

# kitchen 任务 reward × 0.25 → Task Completion Rate
# 正则: 从 run 名中提取 method_key 和 seed
RE_PATTERN = r'(?:kitchen-complete-v0)_(.*?)(?:_seed(\d+)|_(\d+)|-(\d+))(?: - .*)?$'

# x 轴换算参数 (kitchen-complete-v0, n_envs=40, n_steps=200, act_steps=4)
N_PARALLEL_ENVS  = 40
N_ROLLOUT_STEPS  = 200
N_ACT_STEPS      = 4


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_and_aggregate(csv_path: str):
    """
    读取 CSV，按 original_name 分组聚合多 seed，返回:
        {display_name: {'mean': np.array, 'std': np.array, 'steps': np.array}}
    以及颜色映射 {display_name: color}
    """
    data = pd.read_csv(csv_path)

    # kitchen: 乘以 0.25 转换为 Task Completion Rate（0-1）
    for col in data.columns:
        if col != 'Step':
            data[col] = data[col] * 0.25

    # original_name → display_name / color / is_ours 快速查找表
    cfg_lookup = {c['original_name']: c for c in METHOD_CONFIG}

    # 先按 original_name 收集各 seed 的列
    method_seed_cols: dict[str, list] = {}   # original_name → [(seed, col), ...]
    for col in data.columns:
        if col == 'Step':
            continue
        if '__MIN' in col or '__MAX' in col:
            continue
        if 'avg episode reward' not in col.lower() and 'success rate' not in col.lower():
            continue
        m = re.search(RE_PATTERN, col)
        if not m:
            continue
        original_name = m.group(1)
        seed_str = m.group(2) or m.group(3) or m.group(4) or '0'
        seed = int(seed_str)
        method_seed_cols.setdefault(original_name, []).append((seed, col))

    # 只保留 METHOD_CONFIG 中定义的方法
    method_stats: dict[str, dict] = {}
    color_map:    dict[str, str]  = {}

    for cfg in METHOD_CONFIG:
        orig = cfg['original_name']
        disp = cfg['display_name']
        color_map[disp] = cfg['color']

        if orig not in method_seed_cols:
            print(f"  [WARN] '{orig}' not found in CSV, skipping.")
            continue

        seed_data = []
        step_arrays = []
        for _, col in method_seed_cols[orig]:
            valid = data[col].notna()
            seed_data.append(data.loc[valid, col].values)
            step_arrays.append(data.loc[valid, 'Step'].values)

        if not seed_data:
            continue

        min_len = min(len(r) for r in seed_data)
        arr = np.array([r[:min_len] for r in seed_data])   # (n_seeds, T)

        method_stats[disp] = {
            'mean':  np.nanmean(arr, axis=0),
            'std':   np.nanstd(arr,  axis=0),
            'steps': step_arrays[0][:min_len],
            'n_seeds': len(seed_data),
        }
        print(f"  {disp}: {len(seed_data)} seeds, {min_len} steps  "
              f"(final mean={method_stats[disp]['mean'][-1]:.3f})")

    return data, method_stats, color_map


# ── 绘图 ──────────────────────────────────────────────────────────────────────

def plot_ablation(method_stats, color_map, x_axis='sample',
                  xlim=None, ylim=None, output_dir=None, title=None):
    """
    绘制消融图，风格与 plot_without_zoom.py 保持一致。
    """
    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    for cfg in METHOD_CONFIG:
        disp = cfg['display_name']
        if disp not in method_stats:
            continue

        stats = method_stats[disp]
        mean  = stats['mean'].copy()
        std   = stats['std'].copy()
        steps = stats['steps']
        color = color_map[disp]
        is_ours = cfg['is_ours']

        lw    = 3.0  if is_ours else 1.8
        alpha = 1.0  if is_ours else 0.7

        # x 轴换算
        if x_axis == 'sample':
            x = steps * N_PARALLEL_ENVS * N_ROLLOUT_STEPS * N_ACT_STEPS
            x_label = 'Samples'
        elif x_axis == 'step':
            x = steps
            x_label = 'Steps'
        else:
            raise ValueError(f"Unsupported x_axis: {x_axis}")

        ax.plot(x, mean, linewidth=lw, color=color, alpha=alpha,
                label=disp)

        # 方差带，clamp 到 [0, 1]
        lower = np.maximum(mean - std, 0.0)
        upper = np.minimum(mean + std, 1.0)
        ax.fill_between(x, lower, upper, alpha=0.12, color=color)

    # 坐标轴样式
    ax.set_xlabel(x_label,              fontsize=24)
    ax.set_ylabel('Task Completion Rate', fontsize=24)
    ax.tick_params(axis='both', labelsize=20)
    ax.grid(True)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    if title:
        ax.set_title(title, fontsize=18)

    # 图例
    legend = ax.legend(fontsize=15, framealpha=0.9,
                       loc='lower right', borderpad=0.8)

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{x_axis}"
    base = os.path.join(output_dir, f"lamda_ablation_kitchen_complete{suffix}")
    fig.savefig(f'{base}.png', bbox_inches='tight', dpi=300)
    fig.savefig(f'{base}.pdf', bbox_inches='tight')
    print(f"\nSaved:\n  {base}.png\n  {base}.pdf")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='λ ablation plot for kitchen-complete-v0')
    parser.add_argument(
        '--csv', default=None,
        help='Path to CSV file (default: visualize/Final_experiments/data/kitchen-complete/compare_lamda.csv)'
    )
    parser.add_argument(
        '--x_axis', default='sample', choices=['sample', 'step'],
        help='X-axis type: sample or step (default: sample)'
    )
    parser.add_argument(
        '--xlim', nargs=2, type=float, default=None, metavar=('XMIN', 'XMAX'),
        help='X-axis range, e.g. --xlim 0 5000000'
    )
    parser.add_argument(
        '--ylim', nargs=2, type=float, default=None, metavar=('YMIN', 'YMAX'),
        help='Y-axis range, e.g. --ylim 0 1'
    )
    parser.add_argument(
        '--output_dir', default=None,
        help='Output directory (default: visualize/Final_experiments/outs/ablation/lamda)'
    )
    parser.add_argument(
        '--title', default='Ablation: λ Schedule (kitchen-complete-v0)',
        help='Plot title'
    )
    args = parser.parse_args()

    os.chdir(_PROJECT_ROOT)

    csv_path = args.csv or os.path.join(
        _PROJECT_ROOT,
        'visualize/Final_experiments/data/kitchen-complete/compare_lamda.csv'
    )
    output_dir = args.output_dir or os.path.join(
        _PROJECT_ROOT,
        'visualize/Final_experiments/outs/ablation/lamda'
    )

    print("=" * 60)
    print(f"λ Ablation Plot  |  kitchen-complete-v0")
    print(f"CSV  : {csv_path}")
    print(f"X-axis: {args.x_axis}")
    print("=" * 60)

    data, method_stats, color_map = load_and_aggregate(csv_path)

    plot_ablation(
        method_stats, color_map,
        x_axis=args.x_axis,
        xlim=args.xlim,
        ylim=args.ylim,
        output_dir=output_dir,
        title=args.title,
    )
    print("Done!")


if __name__ == '__main__':
    main()
