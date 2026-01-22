#!/usr/bin/env python3
"""
绘制不同初始噪声 std 的消融实验图

用法:
    python agent/eval/visualize/plot_std_ablation.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 计算项目根目录
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

# ==================== 配置区域 ====================

# CSV 文件路径
CSV_PATH = 'visualize/Final_experiments/data/finetune/gym-state/ant-d4rl/ant-eps-compare.csv'

# 方法配置：根据 wandb 运行的时间戳来区分不同的 std 初值
# 格式: 'timestamp_prefix': {'display_name': 'xxx', 'color': 'xxx'}
# TODO: 请根据实际情况修改这里的配置
METHODS = {
    '2026-01-15_04-05-06': {'display_name': 'std=0.001', 'color': '#9b59b6'},   # 紫色
    '2026-01-15_06-28-12': {'display_name': 'std=0.01', 'color': '#3498db'},   # 蓝色
    '2026-01-15_06-11-38': {'display_name': 'std=0.1', 'color': '#2ecc71'},   # 绿色
    '2026-01-16_03-05-33': {'display_name': 'std=1', 'color': '#e74c3c'},   # 红色
}

# 样本计算参数 (用于 x 轴转换)
N_PARALLEL_ENVS = 50
N_ROLLOUT_STEPS = 1000
N_ACT_STEPS = 4
PLOT_X_AXIS = 'sample'  # 'step' 或 'sample'

# 绘图配置
FIGSIZE = (10, 6)
FONTSIZE = 24
CUSTOM_XLIM = None  # [0, 200] 或 None
CUSTOM_YLIM = None  # [0, 5000] 或 None
SHOW_LEGEND = True

# 输出配置
OUTPUT_DIR = 'visualize/Final_experiments/outs/gym-state/ant-d4rl'
OUTPUT_FILENAME = 'std_ablation'

# ==================== 主函数 ====================

def main():
    csv_path = os.path.join(_PROJECT_ROOT, CSV_PATH)
    print(f"Reading CSV from: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    data = pd.read_csv(csv_path)
    print(f"Loaded {len(data)} rows, {len(data.columns)} columns")
    
    # 解析列名，匹配方法
    method_cols = {}  # {display_name: [col1, col2, ...]}
    
    for col in data.columns:
        if col == 'Step':
            continue
        if '__MIN' in col or '__MAX' in col:
            continue
        if 'episode reward' not in col.lower() and 'success rate' not in col.lower():
            continue
        
        # 匹配时间戳前缀
        for timestamp, config in METHODS.items():
            if timestamp in col:
                display_name = config['display_name']
                if display_name not in method_cols:
                    method_cols[display_name] = []
                method_cols[display_name].append(col)
                break
    
    print(f"Found methods: {list(method_cols.keys())}")
    for name, cols in method_cols.items():
        print(f"  {name}: {len(cols)} runs")
    
    # 计算每个方法的统计量
    method_stats = {}
    for display_name, cols in method_cols.items():
        # 获取有效数据的交集索引
        valid_indices_list = [set(data[col].dropna().index) for col in cols]
        common_indices = sorted(list(set.intersection(*valid_indices_list)))
        
        if not common_indices:
            print(f"Warning: {display_name} has no common valid indices, skipping")
            continue
        
        rates = np.array([data.loc[common_indices, col].values for col in cols])
        method_stats[display_name] = {
            'mean': np.nanmean(rates, axis=0),
            'std': np.nanstd(rates, axis=0),
            'steps': data.loc[common_indices, 'Step'].values
        }
    
    # 计算 x 轴
    x_label = 'Steps' if PLOT_X_AXIS == 'step' else 'Samples'
    
    # 设置绘图样式
    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=FIGSIZE)
    handles, labels = [], []
    
    # 绘制每个方法
    for display_name, stats in method_stats.items():
        config = next((v for k, v in METHODS.items() if v['display_name'] == display_name), None)
        color = config['color'] if config else 'gray'
        
        mean = stats['mean']
        std = stats['std']
        steps = stats['steps']
        
        if PLOT_X_AXIS == 'sample':
            x = steps * N_PARALLEL_ENVS * N_ROLLOUT_STEPS * N_ACT_STEPS
        else:
            x = steps
        
        line, = ax.plot(x, mean, label=display_name, linewidth=3, color=color)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)
        handles.append(line)
        labels.append(display_name)
        print(f"Plotted {display_name}: {len(mean)} points, final mean={mean[-1]:.1f}")
    
    # 设置标签
    ax.set_xlabel(x_label, fontsize=FONTSIZE)
    ax.set_ylabel('Average Episode Reward', fontsize=FONTSIZE)
    ax.tick_params(axis='both', labelsize=FONTSIZE - 4)
    ax.grid(True)
    
    if CUSTOM_XLIM: ax.set_xlim(CUSTOM_XLIM)
    if CUSTOM_YLIM: ax.set_ylim(CUSTOM_YLIM)
    LEGEND_ORDER = ['std=0.001', 'std=0.01', 'std=0.1', 'std=1']
    # 按 LEGEND_ORDER 排序图例
    if SHOW_LEGEND:
        if LEGEND_ORDER:
            # 创建排序后的 handles 和 labels
            label_to_handle = dict(zip(labels, handles))
            sorted_handles = []
            sorted_labels = []
            for name in LEGEND_ORDER:
                if name in label_to_handle:
                    sorted_handles.append(label_to_handle[name])
                    sorted_labels.append(name)
            ax.legend(sorted_handles, sorted_labels, fontsize=18, loc='lower right')
        else:
            ax.legend(handles, labels, fontsize=18, loc='lower right')
    
    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    fig.savefig(f'{output_path}.png', bbox_inches='tight', dpi=300)
    fig.savefig(f'{output_path}.pdf', bbox_inches='tight')
    print(f"\nSaved to:\n  - {output_path}.png\n  - {output_path}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()

