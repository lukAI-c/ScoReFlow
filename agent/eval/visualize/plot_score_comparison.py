#!/usr/bin/env python3
"""
绘制 Kitchen-Complete 任务对比图
对比 ReinFlow-R / ReinFlow-S 和 Score-based SDE 方法

用法:
    python agent/eval/visualize/plot_score_comparison.py
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

# 方法配置: original_name 是 wandb 中的名称（CSV 列名中的关键字）
METHODS = [
    {
        'original_name': 'ppo_reflow_mlp_ta4_td4_tdf4',
        'display_name': 'ReinFlow-R',
        'color': '#2ecc71',  # 绿色
    },
    {
        'original_name': 'ppo_reflow_mlp_score_ta4_td4_gamma1',
        'display_name': 'Score-based SDE (ours)',
        'color': '#e74c3c',  # 红色
    },
    # 如果有 ReinFlow-R，取消下面的注释并修改 original_name
    # {
    #     'original_name': 'ppo_reflow_mlp_ta4_td4',
    #     'display_name': 'ReinFlow-R',
    #     'color': '#3498db',  # 蓝色
    # },
]

# 数据配置
CSV_PATH = 'visualize/Final_experiments/data/finetune/kitchen/kitchen-complete-v0/reinflow-randscore.csv'
# 如果你的 CSV 在其他位置，修改上面的路径

# 正则表达式匹配列名中的方法和种子
import re
RE_PATTERN = r'(?:kitchen-complete-v0)_(.*?)(?:_seed(\d+)|_(\d+)|-(\d+))(?: - .*)?$'

# 样本计算参数
N_PARALLEL_ENVS = 40
N_ROLLOUT_STEPS = 200
N_ACT_STEPS = 4

# 绘图配置
FIGSIZE = (10, 6)
FONTSIZE = 24
CUSTOM_XLIM = [0, 32000000]  # X 轴范围 (Samples)
CUSTOM_YLIM = [0, 1.0]     # Y 轴范围 (Task Completion Rate)
SHOW_LEGEND = True

# 输出配置
OUTPUT_DIR = 'visualize/Final_experiments/outs/kitchen/kitchen-complete-v0'
OUTPUT_FILENAME = 'score_comparison'

# ==================== 主函数 ====================

def main():
    # 读取 CSV
    csv_path = os.path.join(_PROJECT_ROOT, CSV_PATH)
    print(f"Reading CSV from: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}\n请先导出 wandb 数据到该路径")
    
    data = pd.read_csv(csv_path)
    
    # Kitchen 环境：成功率乘以 0.25 得到 Task Completion Rate
    data.loc[:, data.columns != 'Step'] = data.loc[:, data.columns != 'Step'] * 0.25
    
    # 计算 x 轴 (Samples)
    raw_steps = data['Step'].values
    x_axis = raw_steps * N_PARALLEL_ENVS * N_ROLLOUT_STEPS * N_ACT_STEPS
    
    # 解析列名，提取方法和种子
    method_seed_map = {}
    for col in data.columns:
        if ('success rate' in col.lower() or 'episode reward' in col.lower()) and '__MIN' not in col and '__MAX' not in col:
            match = re.search(RE_PATTERN, col)
            if match:
                method = match.group(1)
                seed = int(match.group(2) or match.group(3) or match.group(4) or 0)
                if method not in method_seed_map:
                    method_seed_map[method] = []
                method_seed_map[method].append((seed, col))
    
    print(f"Found methods in CSV: {list(method_seed_map.keys())}")
    
    # 设置绘图样式
    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    handles, labels = [], []
    
    # 绘制每个方法
    for method_cfg in METHODS:
        original_name = method_cfg['original_name']
        display_name = method_cfg['display_name']
        color = method_cfg['color']
        
        if original_name not in method_seed_map:
            print(f"Warning: {original_name} not found in CSV, skipping")
            continue
        
        # 收集所有种子的数据
        seed_cols = method_seed_map[original_name]
        suc_rates = []
        for seed, col in seed_cols:
            rate = data[col].dropna().values
            suc_rates.append(rate)
        
        if not suc_rates:
            continue
        
        # 对齐长度
        min_len = min(len(r) for r in suc_rates)
        truncated = [r[:min_len] for r in suc_rates]
        suc_rates = np.array(truncated)
        
        mean = np.nanmean(suc_rates, axis=0)
        std = np.nanstd(suc_rates, axis=0)
        
        # 判断是否是 "ours" 方法
        is_ours = '(ours)' in display_name
        linewidth = 4 if is_ours else 2
        alpha = 1.0 if is_ours else 0.7
        
        # 绘制曲线
        method_x = x_axis[:len(mean)]
        line, = ax.plot(method_x, mean, label=display_name, linewidth=linewidth, color=color, alpha=alpha)
        ax.fill_between(method_x, mean - std, mean + std, alpha=0.15, color=color)
        handles.append(line)
        labels.append(display_name)
        
        print(f"Plotted {display_name}: {len(seed_cols)} seeds, final mean={mean[-1]:.3f}")
    
    # 设置标签
    ax.set_xlabel('Samples', fontsize=FONTSIZE)
    ax.set_ylabel('Task Completion Rate', fontsize=FONTSIZE)
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.grid(True)
    
    # 应用自定义范围
    if CUSTOM_XLIM:
        ax.set_xlim(CUSTOM_XLIM[0], CUSTOM_XLIM[1])
    if CUSTOM_YLIM:
        ax.set_ylim(CUSTOM_YLIM[0], CUSTOM_YLIM[1])
    
    if SHOW_LEGEND:
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

