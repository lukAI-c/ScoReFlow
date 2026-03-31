#!/usr/bin/env python3
"""
α 消融实验绘图脚本 - 统一支持多任务

比较: α=0 (no score guidance) vs α learned (GammaNet, ours)

用法:
    # Humanoid (默认)
    python agent/eval/visualize/plot_alpha_ablation.py --task humanoid

    # Kitchen-complete
    python agent/eval/visualize/plot_alpha_ablation.py --task kitchen

    # 指定参数
    python agent/eval/visualize/plot_alpha_ablation.py --task humanoid --x_axis step --ylim 0 6000
    python agent/eval/visualize/plot_alpha_ablation.py --task kitchen --xlim 0 8000000 --ylim 0 1
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

# ── 各任务配置 ──────────────────────────────────────────────────────────────
TASK_CONFIGS = {
    'humanoid': {
        'csv_default': 'visualize/Final_experiments/data/ablation/alpha/humaniod/all_methods_alpha.csv',
        're_pattern': r'(?:Humanoid-medium-v3)_(.*?)(?:_seed(\d+)|_(\d+)|-(\d+))(?: - .*)?$',
        'metric_key': 'avg episode reward',
        'y_label': 'Average Episode Reward',
        'is_kitchen': False,
        'n_parallel_envs': 40,
        'n_rollout_steps': 500,
        'n_act_steps': 4,
        'methods': [
            {
                'original_name': 'ppo_reflow_mlp_with_score_gammanet_const_alpha0.0_ta4_td4_tdf4',
                'display_name':  r'$\alpha=0$ (w/o score)',
                'color':         '#808080',
                'is_ours':       False,
            },
            {
                'original_name': 'ppo_reflow_mlp_with_score_gammanet_const_alpha1.0_ta4_td4_tdf4',
                'display_name':  r'$\alpha=1$ (constant)',
                'color':         '#E8541A',
                'is_ours':       False,
            },
            {
                'original_name': 'ppo_reflow_mlp_with_score_gammanet_ta4_td4',
                'display_name':  r'$\alpha$ learned (ours)',
                'color':         '#4335FF',
                'is_ours':       True,
            },
        ],
    },
    'kitchen': {
        'csv_default': 'visualize/Final_experiments/data/ablation/alpha/kitchen-complete/all_methods_merged_alpha.csv',
        're_pattern': r'(?:kitchen-complete-v0)_(.*?)(?:_seed(\d+)|_(\d+)|-(\d+))(?: - .*)?$',
        'metric_key': 'avg episode reward',
        'y_label': 'Task Completion Rate',
        'is_kitchen': True,
        'n_parallel_envs': 40,
        'n_rollout_steps': 200,
        'n_act_steps': 4,
        'methods': [
            {
                'original_name': 'ppo_shortcut_mlp_with_score_gammanet_const_alpha0.0_ta4_td4_tdf4',
                'display_name':  r'$\alpha=0$ (w/o score)',
                'color':         '#808080',
                'is_ours':       False,
            },
            {
                'original_name': 'ppo_shortcut_mlp_with_score_gammanet_const_alpha1.0_ta4_td4_tdf4',
                'display_name':  r'$\alpha=1$ (constant)',
                'color':         '#E8541A',
                'is_ours':       False,
            },
            {
                'original_name': 'ppo_shortcut_mlp_with_score_gammanet_ta4_td4_tdf4',
                'display_name':  r'$\alpha$ learned (ours)',
                'color':         '#4335FF',
                'is_ours':       True,
            },
        ],
    },
}


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_and_aggregate(csv_path: str, task_cfg: dict):
    data = pd.read_csv(csv_path)
    methods = task_cfg['methods']
    re_pattern = task_cfg['re_pattern']
    metric_key = task_cfg['metric_key']
    is_kitchen = task_cfg['is_kitchen']

    # kitchen: reward × 0.25 → Task Completion Rate
    if is_kitchen:
        for col in data.columns:
            if col != 'Step':
                data[col] = data[col] * 0.25

    # 按 original_name 收集各 seed 的列
    method_seed_cols: dict[str, list] = {}
    for col in data.columns:
        if col == 'Step':
            continue
        if '__MIN' in col or '__MAX' in col:
            continue
        if metric_key not in col.lower():
            continue
        m = re.search(re_pattern, col)
        if not m:
            continue
        original_name = m.group(1)
        seed_str = m.group(2) or m.group(3) or m.group(4) or '0'
        seed = int(seed_str)
        method_seed_cols.setdefault(original_name, []).append((seed, col))

    print(f"  Found methods in CSV: {list(method_seed_cols.keys())}")

    method_stats: dict[str, dict] = {}
    color_map: dict[str, str] = {}

    for cfg in methods:
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
        arr = np.array([r[:min_len] for r in seed_data])

        method_stats[disp] = {
            'mean':  np.nanmean(arr, axis=0),
            'std':   np.nanstd(arr, axis=0),
            'steps': step_arrays[0][:min_len],
            'n_seeds': len(seed_data),
        }
        print(f"  {disp}: {len(seed_data)} seeds, {min_len} steps  "
              f"(final mean={method_stats[disp]['mean'][-1]:.3f})")

    return data, method_stats, color_map


# ── 绘图 ──────────────────────────────────────────────────────────────────────

def plot_ablation(method_stats, color_map, task_cfg: dict, x_axis='sample',
                  xlim=None, ylim=None, output_dir=None, title=None, task_name='humanoid'):
    methods = task_cfg['methods']
    is_kitchen = task_cfg['is_kitchen']
    n_envs = task_cfg['n_parallel_envs']
    n_rollout = task_cfg['n_rollout_steps']
    n_act = task_cfg['n_act_steps']
    y_label = task_cfg['y_label']

    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    for cfg in methods:
        disp = cfg['display_name']
        if disp not in method_stats:
            continue

        stats = method_stats[disp]
        mean  = stats['mean'].copy()
        std   = stats['std'].copy()
        steps = stats['steps']
        color = color_map[disp]
        is_ours = cfg['is_ours']

        lw    = 3.0 if is_ours else 1.8
        alpha = 1.0 if is_ours else 0.7

        if x_axis == 'sample':
            x = steps * n_envs * n_rollout * n_act
            x_label = 'Samples'
        elif x_axis == 'step':
            x = steps
            x_label = 'Steps'
        else:
            raise ValueError(f"Unsupported x_axis: {x_axis}")

        ax.plot(x, mean, linewidth=lw, color=color, alpha=alpha, label=disp)

        if is_kitchen:
            lower = np.maximum(mean - std, 0.0)
            upper = np.minimum(mean + std, 1.0)
        else:
            lower = mean - std
            upper = mean + std
        ax.fill_between(x, lower, upper, alpha=0.12, color=color)

    ax.set_xlabel(x_label, fontsize=24)
    ax.set_ylabel(y_label, fontsize=24)
    ax.tick_params(axis='both', labelsize=20)
    ax.grid(True)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title, fontsize=18)

    ax.legend(fontsize=15, framealpha=0.9, loc='lower right', borderpad=0.8)

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{x_axis}"
    base = os.path.join(output_dir, f"alpha_ablation_{task_name}{suffix}")
    fig.savefig(f'{base}.png', bbox_inches='tight', dpi=300)
    fig.savefig(f'{base}.pdf', bbox_inches='tight')
    print(f"\nSaved:\n  {base}.png\n  {base}.pdf")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='α ablation plot (Humanoid / Kitchen)')
    parser.add_argument(
        '--task', default='humanoid', choices=list(TASK_CONFIGS.keys()),
        help='Task to plot: humanoid or kitchen (default: humanoid)'
    )
    parser.add_argument('--csv', default=None, help='Path to CSV file')
    parser.add_argument('--x_axis', default='sample', choices=['sample', 'step'])
    parser.add_argument('--xlim', nargs=2, type=float, default=None, metavar=('XMIN', 'XMAX'))
    parser.add_argument('--ylim', nargs=2, type=float, default=None, metavar=('YMIN', 'YMAX'))
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--title', default=None)
    args = parser.parse_args()

    os.chdir(_PROJECT_ROOT)

    task_cfg = TASK_CONFIGS[args.task]
    csv_path = args.csv or os.path.join(_PROJECT_ROOT, task_cfg['csv_default'])
    output_dir = args.output_dir or os.path.join(
        _PROJECT_ROOT, 'visualize/Final_experiments/outs/ablation/alpha'
    )

    print("=" * 60)
    print(f"α Ablation Plot  |  {args.task}")
    print(f"CSV  : {csv_path}")
    print(f"X-axis: {args.x_axis}")
    print("=" * 60)

    data, method_stats, color_map = load_and_aggregate(csv_path, task_cfg)

    plot_ablation(
        method_stats, color_map, task_cfg,
        x_axis=args.x_axis,
        xlim=args.xlim,
        ylim=args.ylim,
        output_dir=output_dir,
        title=args.title,
        task_name=args.task,
    )
    print("Done!")


if __name__ == '__main__':
    main()
