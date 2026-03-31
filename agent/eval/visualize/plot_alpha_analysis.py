#!/usr/bin/env python3
"""
α(t) 曲线分析绘图脚本

数据来源: visualize/Final_experiments/data/ablation/alpha/alpha_visual/
  - 4 个 wandb CSV，分别对应 t=0.00, 0.25, 0.50, 0.75 的 alpha_t 随训练步数的变化
  - 每个 CSV 含多个 run（多 seed），自动聚合均值 ± 标准差

输出双图:
  A. α(t) 随 RL 训练步数的收敛曲线（各时间步）
  B. 训练初期 vs 收敛后，α(t) 关于流匹配时间步 t 的分布

用法:
    python agent/eval/visualize/plot_alpha_analysis.py
    python agent/eval/visualize/plot_alpha_analysis.py --data_dir your/custom/path
    python agent/eval/visualize/plot_alpha_analysis.py --step_final 200
"""

import os
import sys
import re
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline

# ── 项目根目录 ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, _PROJECT_ROOT)

# ── 默认数据目录 ────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = 'visualize/Final_experiments/data/ablation/alpha/alpha_visual'
DEFAULT_OUTPUT_DIR = 'visualize/Final_experiments/outs/ablation/alpha'

# ── 时间步配置（从列名解析）────────────────────────────────────────────────
# step_00_t=0.00, step_01_t=0.25, step_02_t=0.50, step_03_t=0.75
TIMESTEP_COLORS = {
    0.00: '#1f77b4',   # 蓝
    0.25: '#ff7f0e',   # 橙
    0.50: '#2ca02c',   # 绿
    0.75: '#d62728',   # 红
}
TIMESTEP_LABELS = {
    0.00: r'$t=0.00$ (Start)',
    0.25: r'$t=0.25$ (Early)',
    0.50: r'$t=0.50$ (Mid)',
    0.75: r'$t=0.75$ (Late)',
}


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def parse_t_value(csv_path: str) -> float:
    """从 CSV 列名中解析 t 值，如 step_01_t=0.25 → 0.25"""
    df_header = pd.read_csv(csv_path, nrows=0)
    for col in df_header.columns:
        m = re.search(r't=(\d+\.\d+)', col)
        if m:
            return float(m.group(1))
    raise ValueError(f"Cannot parse t value from columns in {csv_path}")


def load_csv_aggregate(csv_path: str) -> pd.DataFrame:
    """
    读取单个 wandb CSV，聚合所有 run 的均值和标准差。
    返回: DataFrame with columns [Step, mean, std]
    """
    df = pd.read_csv(csv_path)
    df['Step'] = df['Step'].astype(int)

    # 只取非 MIN/MAX 的数据列
    value_cols = [c for c in df.columns
                  if c != 'Step' and '__MIN' not in c and '__MAX' not in c]

    if not value_cols:
        raise ValueError(f"No valid value columns in {csv_path}")

    vals = df[value_cols].values.astype(float)
    result = pd.DataFrame({
        'Step': df['Step'].values,
        'mean': np.nanmean(vals, axis=1),
        'std':  np.nanstd(vals,  axis=1),
    })
    return result


def load_all_data(data_dir: str) -> dict:
    """
    加载目录下所有 wandb CSV，返回:
        {t_value: DataFrame(Step, mean, std)}
    按 t_value 排序
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, 'wandb_export_*.csv')))
    if not csv_files:
        raise FileNotFoundError(f"No wandb_export_*.csv found in {data_dir}")

    data = {}
    for fpath in csv_files:
        t_val = parse_t_value(fpath)
        df = load_csv_aggregate(fpath)
        data[t_val] = df
        print(f"  t={t_val:.2f}: {len(df)} steps, "
              f"mean range [{df['mean'].min():.4f}, {df['mean'].max():.4f}]")

    return dict(sorted(data.items()))


# ── 绘图 ──────────────────────────────────────────────────────────────────────

def plot_alpha_analysis(data: dict, step_init: int, step_final: int,
                        output_dir: str):
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.2)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.linewidth'] = 1.2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Plot A: α(t) 随训练步收敛曲线 ──────────────────────────────────────
    for t_val, df in data.items():
        color = TIMESTEP_COLORS.get(t_val, 'gray')
        label = TIMESTEP_LABELS.get(t_val, f't={t_val:.2f}')
        ax1.plot(df['Step'], df['mean'], label=label, color=color, linewidth=2.5)
        ax1.fill_between(df['Step'],
                         df['mean'] - df['std'],
                         df['mean'] + df['std'],
                         alpha=0.15, color=color)

    ax1.set_title(r'A. Convergence of $\alpha_\psi(t)$ during RL Finetuning',
                  fontsize=14, pad=10, weight='bold')
    ax1.set_xlabel('RL Finetuning Step', fontsize=12, weight='bold')
    ax1.set_ylabel(r'Score Multiplier $\alpha_\psi(t)$', fontsize=12, weight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # ── Plot B: 初始 vs 收敛后的 t 分布 ─────────────────────────────────────
    t_vals = sorted(data.keys())

    def get_alpha_at_step(df: pd.DataFrame, step: int) -> float:
        row = df[df['Step'] == step]
        if not row.empty:
            return float(row['mean'].values[0])
        # 找最近的 step
        idx = (df['Step'] - step).abs().argmin()
        return float(df['mean'].iloc[idx])

    val_init  = np.array([get_alpha_at_step(data[t], step_init)  for t in t_vals])
    val_final = np.array([get_alpha_at_step(data[t], step_final) for t in t_vals])

    # 加入 t=1.0（物理约束 α→0）
    t_plot        = np.append(t_vals, 1.0)
    val_init_plot  = np.append(val_init,  0.0)
    val_final_plot = np.append(val_final, 0.0)

    # 样条插值
    t_smooth = np.linspace(t_plot.min(), t_plot.max(), 300)
    k = min(2, len(t_plot) - 1)
    spl_init  = make_interp_spline(t_plot, val_init_plot,  k=k)
    spl_final = make_interp_spline(t_plot, val_final_plot, k=k)

    ax2.plot(t_smooth, spl_init(t_smooth),
             label=f'Step {step_init} (Initial)',
             color='#9467bd', linewidth=2.5, linestyle='--')
    ax2.plot(t_smooth, spl_final(t_smooth),
             label=f'Step {step_final} (Converged)',
             color='#d62728', linewidth=2.5)
    ax2.scatter(t_plot, val_init_plot,  color='#9467bd', s=80, zorder=5)
    ax2.scatter(t_plot, val_final_plot, color='#d62728', s=80, zorder=5)

    ax2.set_title(r'B. $\alpha_\psi(t)$ Profile: Initial vs. Converged',
                  fontsize=14, pad=10, weight='bold')
    ax2.set_xlabel('Flow Matching Timestep $t$', fontsize=12, weight='bold')
    ax2.set_ylabel(r'Score Multiplier $\alpha_\psi(t)$', fontsize=12, weight='bold')
    ax2.set_xticks(sorted(set(list(t_vals) + [1.0])))
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 加总标题标注任务
    fig.suptitle('α(t) Analysis - Kitchen-Complete-v0 Task',
                 fontsize=16, weight='bold', y=0.98)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, 'alpha_analysis_plot')
    fig.savefig(f'{base}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{base}.pdf', bbox_inches='tight')
    print(f"\nSaved:\n  {base}.png\n  {base}.pdf")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='α(t) analysis plot')
    parser.add_argument(
        '--data_dir', default=None,
        help=f'Directory with wandb CSV files (default: {DEFAULT_DATA_DIR})'
    )
    parser.add_argument(
        '--output_dir', default=None,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--step_init', type=int, default=1,
        help='Initial RL step for Plot B (default: 1)'
    )
    parser.add_argument(
        '--step_final', type=int, default=None,
        help='Final RL step for Plot B (default: last step in data)'
    )
    args = parser.parse_args()

    os.chdir(_PROJECT_ROOT)

    data_dir   = args.data_dir   or os.path.join(_PROJECT_ROOT, DEFAULT_DATA_DIR)
    output_dir = args.output_dir or os.path.join(_PROJECT_ROOT, DEFAULT_OUTPUT_DIR)

    print("=" * 60)
    print(f"α(t) Analysis Plot")
    print(f"Data dir : {data_dir}")
    print("=" * 60)

    data = load_all_data(data_dir)

    # 自动确定 step_final
    if args.step_final is None:
        all_steps = sorted(set(s for df in data.values() for s in df['Step']))
        step_final = all_steps[-1]
        print(f"  Auto step_final = {step_final}")
    else:
        step_final = args.step_final

    plot_alpha_analysis(data, args.step_init, step_final, output_dir)
    print("Done!")


if __name__ == '__main__':
    main()
