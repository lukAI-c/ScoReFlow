#!/usr/bin/env python3
"""
分析不同去噪步数对 Success Rate 的影响
用法:
python agent/eval/visualize/analyze_denoise_steps.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目路径
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, _PROJECT_ROOT)

# 数据路径
CSV_PATH = os.path.join(_PROJECT_ROOT, 'visualize/Final_experiments/data/finetune/robomimic-img/square-img-denoise-steps/compare-steps.csv')

def analyze_denoise_steps():
    """分析不同去噪步数的影响"""
    
    print("=" * 80)
    print("分析不同去噪步数对 Success Rate 的影响")
    print("=" * 80)
    print()
    
    # 加载数据
    data = pd.read_csv(CSV_PATH)
    print(f"✓ 加载数据: {CSV_PATH}")
    print(f"  数据形状: {data.shape}")
    print()
    
    # 识别不同去噪步数的列
    denoise_steps_map = {
        '1 step': [],
        '2 steps': [],
        '4 steps': []
    }
    
    for col in data.columns:
        if 'success rate' in col.lower() and '__MIN' not in col and '__MAX' not in col:
            if 'td1_tdf1' in col:
                denoise_steps_map['1 step'].append(col)
            elif 'td2_tdf1' in col:
                denoise_steps_map['2 steps'].append(col)
            elif 'td4_tdf1' in col:
                denoise_steps_map['4 steps'].append(col)
    
    print("识别的去噪步数配置:")
    for steps, cols in denoise_steps_map.items():
        print(f"  {steps}: {len(cols)} seeds")
        for col in cols:
            print(f"    - {col[:80]}...")
    print()
    
    # 计算每个配置的统计数据
    stats = {}
    for steps, cols in denoise_steps_map.items():
        if not cols:
            continue
        
        # 提取所有 seed 的数据
        values = []
        for col in cols:
            values.append(data[col].dropna().values)
        
        # 对齐长度
        min_len = min(len(v) for v in values)
        values = [v[:min_len] for v in values]
        values = np.array(values)
        
        # 计算统计
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        
        stats[steps] = {
            'mean': mean,
            'std': std,
            'n_seeds': len(cols),
            'steps': data['Step'].values[:min_len]
        }
    
    # 打印统计结果
    print("=" * 80)
    print("统计结果")
    print("=" * 80)
    print()
    
    for steps in ['1 step', '2 steps', '4 steps']:
        if steps not in stats:
            continue
        
        mean = stats[steps]['mean']
        std = stats[steps]['std']
        n_seeds = stats[steps]['n_seeds']
        
        print(f"{steps}:")
        print(f"  Seeds: {n_seeds}")
        print(f"  初始 Success Rate: {mean[0]*100:.2f}% ± {std[0]*100:.2f}%")
        print(f"  最终 Success Rate: {mean[-1]*100:.2f}% ± {std[-1]*100:.2f}%")
        print(f"  提升: {(mean[-1] - mean[0])*100:.2f}%")
        print(f"  提升比例: {mean[-1]/mean[0]*100:.2f}%")
        print()
    
    # 对比分析
    print("=" * 80)
    print("对比分析")
    print("=" * 80)
    print()
    
    if '1 step' in stats and '2 steps' in stats:
        mean_1 = stats['1 step']['mean'][-1]
        mean_2 = stats['2 steps']['mean'][-1]
        print(f"2 steps vs 1 step:")
        print(f"  最终 Success Rate: {mean_2*100:.2f}% vs {mean_1*100:.2f}%")
        print(f"  差异: {(mean_2 - mean_1)*100:+.2f}%")
        print(f"  相对提升: {(mean_2/mean_1 - 1)*100:+.2f}%")
        print()
    
    if '2 steps' in stats and '4 steps' in stats:
        mean_2 = stats['2 steps']['mean'][-1]
        mean_4 = stats['4 steps']['mean'][-1]
        print(f"4 steps vs 2 steps:")
        print(f"  最终 Success Rate: {mean_4*100:.2f}% vs {mean_2*100:.2f}%")
        print(f"  差异: {(mean_4 - mean_2)*100:+.2f}%")
        print(f"  相对提升: {(mean_4/mean_2 - 1)*100:+.2f}%")
        print()
    
    if '1 step' in stats and '4 steps' in stats:
        mean_1 = stats['1 step']['mean'][-1]
        mean_4 = stats['4 steps']['mean'][-1]
        print(f"4 steps vs 1 step:")
        print(f"  最终 Success Rate: {mean_4*100:.2f}% vs {mean_1*100:.2f}%")
        print(f"  差异: {(mean_4 - mean_1)*100:+.2f}%")
        print(f"  相对提升: {(mean_4/mean_1 - 1)*100:+.2f}%")
        print()
    
    return stats


if __name__ == '__main__':
    stats = analyze_denoise_steps()
    print("✅ 分析完成！")
    print()
    print("提示: 运行以下命令生成可视化图表:")
    print("  python agent/eval/visualize/success_rate_episode_reward.py --config-name=square_denoise_steps")

