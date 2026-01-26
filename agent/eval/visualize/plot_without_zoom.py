#!/usr/bin/env python3
"""
无放大窗口的绘图脚本 - 只绘制主图
用法:
python agent/eval/visualize/plot_no_zoom.py environment_name=gym-state task_name=humanoid-d4rl plot_x_axis=wallclock
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

    # 计算统计 - 使用 dropna 后的值数组（与 original_print.py 一致）
    method_stats = {}
    for method, seed_cols in updated_map.items():
        # 收集每个 seed 的数据（dropna 后的值）
        suc_rates = []
        step_arrays = []  # 保存每个 seed 对应的 Step 值
        for _, col in seed_cols:
            valid_mask = data[col].notna()
            rate = data.loc[valid_mask, col].values
            steps = data.loc[valid_mask, 'Step'].values
            suc_rates.append(rate)
            step_arrays.append(steps)

        if not suc_rates:
            continue

        # 取最短长度截断（与 original_print.py 一致）
        min_len = min(len(r) for r in suc_rates)
        truncated_rates = [r[:min_len] for r in suc_rates]
        arr = np.array(truncated_rates)

        # 使用第一个 seed 的 Step 值作为 x 轴（截断到 min_len）
        method_steps = step_arrays[0][:min_len]

        # 保存统计数据和对应的 Step 值
        method_stats[method] = {
            'mean': np.nanmean(arr, axis=0),
            'std': np.nanstd(arr, axis=0),
            'steps': method_steps  # 每个方法自己的 Step 值
        }

    # 颜色映射
    color_map = {cfg['display_name']: cfg['color'] for cfg in method_config}
    for m in updated_map:
        if m not in color_map:
            color_map[m] = color_dict.get(m, 'gray')

    return data, method_stats, color_map





@hydra.main(config_path="visualize_cfgs", config_name="final_experiments", version_base="1.1")
def main(cfg: DictConfig):
    """主函数 - 绘制无放大窗口的图"""
    os.chdir(_PROJECT_ROOT)

    # 获取参数
    environment_name = cfg.get('environment_name', 'kitchen')
    task_name = cfg.get('task_name', 'kitchen-complete-v0')
    plot_x_axis = cfg.get('plot_x_axis', 'sample')
    is_wallclock = (plot_x_axis == 'wallclock')

    # 自定义坐标轴范围（可选）
    custom_xlim = list(cfg.get('custom_xlim')) if cfg.get('custom_xlim') else None
    custom_ylim = list(cfg.get('custom_ylim')) if cfg.get('custom_ylim') else None

    print("=" * 60)
    print(f"Plot (No Zoom): {environment_name}/{task_name}")
    print(f"X-axis: {plot_x_axis}")
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

    # 绘制曲线
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
            # FQL 特殊处理：使用 CSV Step 列的最后 N 个值，只乘以 n_act_steps
            # （与 success_rate_episode_reward.py 第 229-232 行一致）
            if method == 'FQL':
                raw_steps = data['Step'].values
                fql_steps = raw_steps[-len(mean):]
                method_x = fql_steps * n_act_steps
            else:
                method_x = method_steps * n_parallel_envs * n_rollout_steps * n_act_steps
        else:
            method_x = method_steps

        ax.plot(method_x, mean, linewidth=lw, color=color, alpha=alpha, label=method)
        # 对 kitchen 任务，限制方差带在 [0, 1] 范围内
        if 'kitchen' in task_name:
            lower_bound = np.maximum(mean - std, 0)
            upper_bound = np.minimum(mean + std, 1)
        else:
            lower_bound = mean - std
            upper_bound = mean + std
        ax.fill_between(method_x, lower_bound, upper_bound, alpha=0.1, color=color)

    # 设置主图样式
    ax.set_xlabel(x_label, fontsize=24)
    ax.set_ylabel(evaluation_name, fontsize=24)
    ax.tick_params(axis='both', labelsize=20)
    ax.grid(True)
    if custom_xlim:
        ax.set_xlim(custom_xlim)
    if custom_ylim:
        ax.set_ylim(custom_ylim)

    # 保存图形
    output_dir = os.path.join(_PROJECT_ROOT, 'visualize/Final_experiments/outs')
    fig_dir = os.path.join(output_dir, environment_name, task_name)
    os.makedirs(fig_dir, exist_ok=True)

    suffix = "_wallclock" if is_wallclock else ""
    output_filename = f"{environment_name}_{task_name}_{evaluation_name}{suffix}_no_zoom"
    output_path = os.path.join(fig_dir, output_filename)

    fig.savefig(f'{output_path}.png', bbox_inches='tight', dpi=300)
    fig.savefig(f'{output_path}.pdf', bbox_inches='tight')
    print(f"Saved: {output_path}.png / .pdf")
    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    main()

