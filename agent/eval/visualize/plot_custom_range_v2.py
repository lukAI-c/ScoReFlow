#!/usr/bin/env python3
"""
自定义 x 轴和 y 轴区间的绘图脚本
完全独立实现，支持自定义 xlim 和 ylim
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
from agent.eval.visualize.constants import method_name_dict, max_n_step_dict



def plot_with_custom_range(
    evaluation_name,
    environment_name,
    task_name,
    re_expression,
    csv_filename,
    output_dir,
    output_filename,
    plot_x_axis,
    n_parallel_envs,
    n_rollout_steps,
    n_act_steps,
    custom_xlim=None,
    custom_ylim=None
):
    """绘图函数，支持自定义 xlim 和 ylim"""

    # 读取 CSV 文件
    csv_path = os.path.join(_PROJECT_ROOT, f'visualize/Final_experiments/data/finetune/{environment_name}/{task_name}/{csv_filename}')
    print(f"Reading CSV from: {csv_path}")
    data = pd.read_csv(csv_path)

    if 'kitchen' in task_name:
        evaluation_name = 'TaskCompletionRate'
        data.loc[:, data.columns != 'Step'] = data.loc[:, data.columns != 'Step'] * 0.25

    if task_name not in max_n_step_dict:
        raise NotImplementedError(f"task_name={task_name} not in max_n_step_dict")

    method_config = method_name_dict[task_name]

    # 计算 x 轴
    if plot_x_axis == 'step':
        x_axis = data['Step'].values
        x_label = 'Steps'
    elif plot_x_axis == 'sample':
        raw_steps = data['Step'].values
        x_axis = raw_steps * n_parallel_envs * n_rollout_steps * n_act_steps
        x_label = 'Samples'
    else:
        raise ValueError(f"Unsupported plot_x_axis: {plot_x_axis}")

    # 从列名中提取方法和种子
    method_seed_map = {}
    for col in data.columns:
        if ('success rate' in col or 'episode reward' in col) and '__MIN' not in col and '__MAX' not in col:
            match = re.search(re_expression, col)
            if match:
                method = match.group(1)
                seed = int(match.group(2) or match.group(3) or match.group(4))
                if method not in method_seed_map:
                    method_seed_map[method] = []
                method_seed_map[method].append((seed, col))

    # 更新方法名称
    updated_method_seed_map = {}
    for old_method, seed_cols in method_seed_map.items():
        for config in method_config:
            if config['original_name'] == old_method:
                updated_method_seed_map[config['display_name']] = seed_cols
                break
        else:
            updated_method_seed_map[old_method] = seed_cols

    print(f"Extracted methods: {list(updated_method_seed_map.keys())}")

    # 计算每个方法的统计量
    method_stats = {}
    for method, seed_cols in updated_method_seed_map.items():
        suc_rates = []
        for seed, col in seed_cols:
            rate = data[col].dropna().values
            suc_rates.append(rate)
        min_len = min(len(r) for r in suc_rates)
        truncated = [r[:min_len] for r in suc_rates]
        suc_rates = np.array(truncated)
        method_stats[method] = {
            'mean': np.nanmean(suc_rates, axis=0),
            'std': np.nanstd(suc_rates, axis=0),
            'seeds': [s for s, _ in seed_cols]
        }

    # 设置绘图样式
    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    color_map = {m['display_name']: m['color'] for m in method_config}
    handles, labels = [], []

    # 绘制所有方法
    for method in updated_method_seed_map:
        if method in method_stats:
            stats = method_stats[method]
            mean = stats['mean']
            std = stats['std']
            color = color_map.get(method, 'gray')

            is_ours = '(ours)' in method
            linewidth = 4 if is_ours else 2
            alpha = 1.0 if is_ours else 0.6

            ax.plot(x_axis[:len(mean)], mean, linewidth=linewidth+1, color=color, alpha=alpha)
            line, = ax.plot(x_axis[:len(mean)], mean, label=method, linewidth=linewidth, color=color, alpha=alpha)
            ax.fill_between(x_axis[:len(mean)], mean - std, mean + std, alpha=0.12, color=color)
            handles.append(line)
            labels.append(method)

    # 设置标签
    fontsize = 24
    evaluation_name_spaced = ''.join(' ' + c if c.isupper() else c for c in evaluation_name).strip()
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(evaluation_name_spaced, fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.xaxis.get_offset_text().set_fontsize(24)
    ax.grid(True)

    # 应用自定义区间
    if custom_xlim:
        ax.set_xlim(custom_xlim[0], custom_xlim[1])
        print(f"Applied custom X range: {custom_xlim}")
    if custom_ylim:
        ax.set_ylim(custom_ylim[0], custom_ylim[1])
        print(f"Applied custom Y range: {custom_ylim}")

    # 不添加 legend

    # 保存图片
    fig_dir = os.path.join(output_dir, environment_name, task_name)
    os.makedirs(fig_dir, exist_ok=True)
    output_file_path = os.path.join(fig_dir, output_filename)
    plt.savefig(f'{output_file_path}.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'{output_file_path}.pdf', bbox_inches='tight')
    print(f"Plot saved to:")
    print(f"  - {output_file_path}.png")
    print(f"  - {output_file_path}.pdf")
    plt.close()

    return updated_method_seed_map


@hydra.main(config_path="visualize_cfgs", config_name="final_experiments", version_base="1.1")
def main(cfg: DictConfig):
    """主函数"""

    # Hydra 会切换工作目录，这里需要切回项目根目录
    os.chdir(_PROJECT_ROOT)

    # 获取参数
    evaluation_name = cfg.get('evaluation_name', 'TaskCompletionRate')
    environment_name = cfg.get('environment_name', 'kitchen')
    task_name = cfg.get('task_name', 'kitchen-complete-v0')
    plot_x_axis = cfg.get('plot_x_axis', 'sample')

    # 自定义区间参数
    custom_xlim = cfg.get('custom_xlim', None)
    custom_ylim = cfg.get('custom_ylim', None)

    # 转换为 list（Hydra 可能返回 ListConfig）
    if custom_xlim is not None:
        custom_xlim = list(custom_xlim)
    if custom_ylim is not None:
        custom_ylim = list(custom_ylim)

    print(f"Generating plot for {environment_name}/{task_name}")
    if custom_xlim:
        print(f"Custom X range: {custom_xlim}")
    if custom_ylim:
        print(f"Custom Y range: {custom_ylim}")

    # 获取配置
    env_cfg = cfg.env[environment_name][task_name]
    csv_filename = env_cfg.get('csv_filename', 'all_methods_merged.csv')
    re_expression_raw = env_cfg.get('re_expression', r'')
    n_parallel_envs = env_cfg.get('n_parallel_envs', 1)
    n_rollout_steps = env_cfg.get('n_rollout_steps', 1)
    n_act_steps = env_cfg.get('n_act_steps', 1)

    # 处理 re_expression 格式
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
    output_filename = f'{environment_name}_{task_name}_{evaluation_name}{range_suffix}'

    # 调用绘图函数
    result = plot_with_custom_range(
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
        custom_ylim=custom_ylim
    )

    print(f"\nExtracted methods: {list(result.keys())}")
    print("\nDone!")


if __name__ == "__main__":
    main()

