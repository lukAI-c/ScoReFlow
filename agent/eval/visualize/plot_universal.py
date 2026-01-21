#!/usr/bin/env python3
"""
通用绘图脚本 - 支持 kitchen, gym-state, robomimic-img 环境
支持自定义 xlim 和 ylim，不显示 legend
支持三种 x 轴模式: step, sample, wallclock

用法示例:
    # Kitchen 环境 (Samples)
    python agent/eval/visualize/plot_universal.py \
        environment_name=kitchen task_name=kitchen-complete-v0 \
        plot_x_axis=sample +custom_xlim=[0,2000000] +custom_ylim=[0.6,1]

    # Gym 环境 (Wall Clock Time)
    python agent/eval/visualize/plot_universal.py \
        environment_name=gym-state task_name=hopper \
        plot_x_axis=wallclock +custom_xlim=[0,10]

    # Gym 环境 (Samples)
    python agent/eval/visualize/plot_universal.py \
        environment_name=gym-state task_name=hopper \
        plot_x_axis=sample +custom_xlim=[0,10000000]

    # Robomimic 环境
    python agent/eval/visualize/plot_universal.py \
        environment_name=robomimic-img task_name=square-img
"""

import os
import sys
import re
import hydra
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from omegaconf import DictConfig

# 计算项目根目录并添加到路径
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, _PROJECT_ROOT)

# 导入常量
from agent.eval.visualize.constants import method_name_dict, max_n_step_dict, color_dict, time_step_ratios

# Wall clock time 计算常量
PPO_EVAL_FREQ = 10
FQL_EVAL_FREQ = 5000
FQL_LOG_FREQ = 200


def plot_universal(
    evaluation_name: str,
    environment_name: str,
    task_name: str,
    re_expression: str,
    csv_filename: str,
    output_dir: str,
    output_filename: str,
    plot_x_axis: str,
    n_parallel_envs: int,
    n_rollout_steps: int,
    n_act_steps: int,
    custom_xlim=None,
    custom_ylim=None,
    show_legend: bool = False,
    figsize: tuple = (10, 6),
    fontsize: int = 24
):
    """
    通用绘图函数，支持所有环境类型
    
    Args:
        evaluation_name: 评估指标名称 (SuccessRate, TaskCompletionRate, AverageEpisodeReward)
        environment_name: 环境名称 (kitchen, gym-state, robomimic-img)
        task_name: 任务名称
        re_expression: 正则表达式用于匹配列名
        csv_filename: CSV 文件名
        output_dir: 输出目录
        output_filename: 输出文件名（不含扩展名）
        plot_x_axis: x轴类型 ('step', 'sample', 'wallclock')
        n_parallel_envs: 并行环境数
        n_rollout_steps: rollout步数
        n_act_steps: action步数
        custom_xlim: 自定义x轴范围 [min, max]
        custom_ylim: 自定义y轴范围 [min, max]
        show_legend: 是否显示图例
        figsize: 图形大小
        fontsize: 字体大小

    Returns:
        dict: 方法名到种子列的映射
    """

    # wallclock 模式标志
    is_wallclock = (plot_x_axis == 'wallclock')
    
    # 读取 CSV 文件
    csv_path = os.path.join(_PROJECT_ROOT, f'visualize/Final_experiments/data/finetune/{environment_name}/{task_name}/{csv_filename}')
    print(f"Reading CSV from: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    data = pd.read_csv(csv_path)
    
    # Kitchen 环境特殊处理：成功率需要乘以 0.25
    if 'kitchen' in task_name:
        evaluation_name = 'TaskCompletionRate'
        data.loc[:, data.columns != 'Step'] = data.loc[:, data.columns != 'Step'] * 0.25

    # 检查任务是否在配置中
    if task_name not in max_n_step_dict:
        print(f"Warning: task_name={task_name} not in max_n_step_dict, using default")
    
    if task_name not in method_name_dict:
        raise ValueError(f"task_name={task_name} not in method_name_dict. Supported tasks: {list(method_name_dict.keys())}")
    
    method_config = method_name_dict[task_name]

    # 计算 x 轴（wallclock 模式在绘图时单独处理每个方法）
    raw_steps = data['Step'].values
    if plot_x_axis == 'step':
        x_axis = raw_steps
        x_label = 'Steps'
    elif plot_x_axis == 'sample':
        x_axis = raw_steps * n_parallel_envs * n_rollout_steps * n_act_steps
        x_label = 'Samples'
    elif plot_x_axis == 'wallclock':
        x_axis = None  # wallclock 模式下每个方法的 x 轴不同，在绘图时单独计算
        x_label = 'Wall-Clock Time (hours)'
        # 检查是否有 time_step_ratios 配置
        if task_name not in time_step_ratios:
            raise ValueError(f"task_name={task_name} not in time_step_ratios. Wallclock mode not supported for this task.")
    else:
        raise ValueError(f"Unsupported plot_x_axis: {plot_x_axis}. Must be 'step', 'sample', or 'wallclock'.")

    # 从列名中提取方法和种子
    method_seed_map = {}
    unmatched_columns = []
    
    for col in data.columns:
        if ('success rate' in col.lower() or 'episode reward' in col.lower()) and '__MIN' not in col and '__MAX' not in col:
            match = re.search(re_expression, col)
            if match:
                method = match.group(1)
                seed = int(match.group(2) or match.group(3) or match.group(4) or 0)
                if method not in method_seed_map:
                    method_seed_map[method] = []
                method_seed_map[method].append((seed, col))
            else:
                unmatched_columns.append(col)
    
    if unmatched_columns:
        print(f"Unmatched columns: {unmatched_columns[:5]}...")  # 只显示前5个

    # 更新方法名称为显示名称
    updated_method_seed_map = {}
    for old_method, seed_cols in method_seed_map.items():
        display_name = old_method  # 默认使用原始名称
        for config in method_config:
            if config['original_name'] == old_method:
                display_name = config['display_name']
                break
        updated_method_seed_map[display_name] = seed_cols

    print(f"Extracted methods: {list(updated_method_seed_map.keys())}")

    # 计算每个方法的统计量（均值和标准差）
    method_stats = {}
    for method, seed_cols in updated_method_seed_map.items():
        suc_rates = []
        for seed, col in seed_cols:
            rate = data[col].dropna().values
            suc_rates.append(rate)
        
        if not suc_rates:
            continue
            
        min_len = min(len(r) for r in suc_rates)
        truncated = [r[:min_len] for r in suc_rates]
        suc_rates = np.array(truncated)
        
        method_stats[method] = {
            'mean': np.nanmean(suc_rates, axis=0),
            'std': np.nanstd(suc_rates, axis=0),
            'seeds': [s for s, _ in seed_cols]
        }

    # 构建颜色映射
    color_map = {}
    for config in method_config:
        color_map[config['display_name']] = config['color']
    # 添加默认颜色
    for method in updated_method_seed_map:
        if method not in color_map:
            color_map[method] = color_dict.get(method, 'gray')

    # 设置绘图样式
    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=figsize)

    handles, labels = [], []
    bc_values = []  # 用于 gym-state 的 BC 基线

    # 绘制所有方法
    for method in updated_method_seed_map:
        if method not in method_stats:
            continue

        stats = method_stats[method]
        mean = stats['mean'].copy()
        std = stats['std'].copy()
        color = color_map.get(method, 'gray')

        # 判断是否是 "ours" 方法，使用更粗的线
        is_ours = '(ours)' in method
        linewidth = 4 if is_ours else 2
        alpha = 1.0 if is_ours else 0.6

        # Wallclock 模式：为每个方法单独计算 x 轴
        if is_wallclock:
            # 检查该方法是否有 time_step_ratios 配置
            if method not in time_step_ratios.get(task_name, {}):
                print(f"Skipping {method}: no time_step_ratios config")
                continue

            time_ratio = time_step_ratios[task_name][method]

            # FQL 需要下采样
            if method == 'FQL':
                mean = mean[::5]
                std = std[::5]
                eval_interval = FQL_EVAL_FREQ
                time_per_itr = time_ratio / FQL_LOG_FREQ
            else:
                eval_interval = PPO_EVAL_FREQ
                time_per_itr = time_ratio
                # 记录 BC 值（第一个点的值）
                if method != 'Gaussian':
                    bc_values.append(mean[0])

            # 计算时间轴（小时）
            data_len = len(mean)
            method_x_axis = np.arange(data_len) * eval_interval * time_per_itr / 3600
        else:
            method_x_axis = x_axis[:len(mean)]

        # 绘制曲线
        ax.plot(method_x_axis, mean, linewidth=linewidth+1, color=color, alpha=alpha)
        line, = ax.plot(method_x_axis, mean, label=method, linewidth=linewidth, color=color, alpha=alpha)
        ax.fill_between(method_x_axis, mean - std, mean + std, alpha=0.12, color=color)
        handles.append(line)
        labels.append(method)

    # Gym-state 环境在 wallclock 模式下添加 BC 基线
    if is_wallclock and environment_name == 'gym-state' and bc_values:
        bc_value = np.array(bc_values).mean()
        bc_line = ax.axhline(y=bc_value, color='black', linestyle='--', label='BC', linewidth=4)
        handles.append(bc_line)
        labels.append('BC')

    # 设置标签
    evaluation_name_spaced = ''.join(' ' + c if c.isupper() else c for c in evaluation_name).strip()
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(evaluation_name_spaced, fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.xaxis.get_offset_text().set_fontsize(fontsize)
    ax.grid(True)

    # 应用自定义区间
    if custom_xlim:
        ax.set_xlim(custom_xlim[0], custom_xlim[1])
        print(f"Applied custom X range: {custom_xlim}")
    if custom_ylim:
        ax.set_ylim(custom_ylim[0], custom_ylim[1])
        print(f"Applied custom Y range: {custom_ylim}")

    # 是否显示图例
    if show_legend:
        ax.legend(handles, labels, fontsize=18, loc='lower right')

    # 保存图片
    fig_dir = os.path.join(output_dir, environment_name, task_name)
    os.makedirs(fig_dir, exist_ok=True)
    output_file_path = os.path.join(fig_dir, output_filename)
    fig.savefig(f'{output_file_path}.png', bbox_inches='tight', dpi=300)
    fig.savefig(f'{output_file_path}.pdf', bbox_inches='tight')
    print(f"Plot saved to:")
    print(f"  - {output_file_path}.png")
    print(f"  - {output_file_path}.pdf")
    plt.close(fig)

    return updated_method_seed_map


@hydra.main(config_path="visualize_cfgs", config_name="final_experiments", version_base="1.1")
def main(cfg: DictConfig):
    """主函数 - 通用绘图入口"""

    # Hydra 会切换工作目录，这里需要切回项目根目录
    os.chdir(_PROJECT_ROOT)

    # 获取基本参数
    evaluation_name = cfg.get('evaluation_name', 'SuccessRate')
    environment_name = cfg.get('environment_name', 'kitchen')
    task_name = cfg.get('task_name', 'kitchen-complete-v0')
    plot_x_axis = cfg.get('plot_x_axis', 'sample')

    # 自定义区间参数
    custom_xlim = cfg.get('custom_xlim', None)
    custom_ylim = cfg.get('custom_ylim', None)
    show_legend = cfg.get('show_legend', False)

    # 转换为 list（Hydra 可能返回 ListConfig）
    if custom_xlim is not None:
        custom_xlim = list(custom_xlim)
    if custom_ylim is not None:
        custom_ylim = list(custom_ylim)

    print("=" * 60)
    print(f"Universal Plot Script")
    print("=" * 60)
    print(f"Environment: {environment_name}")
    print(f"Task: {task_name}")
    print(f"Evaluation: {evaluation_name}")
    print(f"X-axis mode: {plot_x_axis}")
    if custom_xlim:
        print(f"Custom X range: {custom_xlim}")
    if custom_ylim:
        print(f"Custom Y range: {custom_ylim}")
    print("=" * 60)

    # 获取任务配置
    env_cfg = cfg.env[environment_name][task_name]
    csv_filename = env_cfg.get('csv_filename', 'all_methods_merged.csv')
    re_expression_raw = env_cfg.get('re_expression', r'')
    n_parallel_envs = env_cfg.get('n_parallel_envs', 1)
    n_rollout_steps = env_cfg.get('n_rollout_steps', 1)
    n_act_steps = env_cfg.get('n_act_steps', 1)

    # 处理 re_expression 格式（去掉 r'' 包装）
    if re_expression_raw.startswith("r''") and re_expression_raw.endswith("''"):
        re_expression = re_expression_raw[3:-2]
    elif re_expression_raw.startswith("r'") and re_expression_raw.endswith("'"):
        re_expression = re_expression_raw[2:-1]
    else:
        re_expression = re_expression_raw

    # 输出目录
    output_dir = os.path.join(_PROJECT_ROOT, 'visualize/Final_experiments/outs')

    # 生成文件名
    range_suffix = "_custom_range" if (custom_xlim or custom_ylim) else ""
    wallclock_suffix = "_wallclock" if plot_x_axis == 'wallclock' else ""
    output_filename = f'{environment_name}_{task_name}_{evaluation_name}{wallclock_suffix}{range_suffix}'

    # 调用通用绘图函数
    result = plot_universal(
        evaluation_name=evaluation_name,
        environment_name=environment_name,
        task_name=task_name,
        re_expression=re_expression,
        csv_filename=csv_filename,
        output_dir=output_dir,
        output_filename=output_filename,
        plot_x_axis=plot_x_axis,
        n_parallel_envs=n_parallel_envs,
        n_rollout_steps=n_rollout_steps,
        n_act_steps=n_act_steps,
        custom_xlim=custom_xlim,
        custom_ylim=custom_ylim,
        show_legend=show_legend
    )

    print(f"\nExtracted {len(result)} methods: {list(result.keys())}")
    print("\nDone!")


if __name__ == "__main__":
    main()

