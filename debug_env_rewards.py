"""
调试环境奖励和终止信号

这个脚本会：
1. 检查环境是否正确返回奖励
2. 检查环境是否正确发送终止信号
3. 打印详细的环境交互信息
"""

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os

log = logging.getLogger(__name__)


@hydra.main(config_path="cfg/robomimic/finetune/square", config_name="ft_fpo_reflow_mlp_img", version_base=None)
def main(cfg: DictConfig):
    print("\n" + "="*80)
    print("环境奖励和终止信号调试")
    print("="*80)
    
    # 设置渲染后端
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    
    # 导入环境
    from env.gym_utils import make_async

    # 创建环境（只用1个环境便于调试）
    print("\n创建环境...")
    print(f"  - 环境名称: {cfg.env.name}")
    print(f"  - 最大步数: {cfg.env.max_episode_steps}")
    print(f"  - 成功奖励阈值: {cfg.env.best_reward_threshold_for_success}")

    venv = make_async(
        env_name=cfg.env.name,
        num_envs=1,  # 只用1个环境
        asynchronous=False,  # 使用同步模式便于调试
        max_episode_steps=cfg.env.max_episode_steps,
        wrappers=cfg.env.wrappers,
        robomimic_env_cfg_path=cfg.robomimic_env_cfg_path,
        shape_meta=cfg.shape_meta,
        use_image_obs=cfg.env.use_image_obs,
        render=False,
        render_offscreen=False,
        obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim
    )
    
    print("✓ 环境创建成功")
    
    # 重置环境
    print("\n重置环境...")
    obs_venv = venv.reset()
    if isinstance(obs_venv, tuple):
        obs_venv = obs_venv[0]
    
    print(f"✓ 环境重置成功")
    print(f"  - 观测类型: {type(obs_venv)}")
    if isinstance(obs_venv, dict):
        for k, v in obs_venv.items():
            if isinstance(v, np.ndarray):
                print(f"  - {k}: shape={v.shape}, dtype={v.dtype}")
    
    # 运行一个完整的 episode
    print("\n" + "="*80)
    print("运行测试 Episode")
    print("="*80)
    
    total_reward = 0
    step_count = 0
    max_steps = cfg.env.max_episode_steps
    
    # 记录奖励和终止信号
    rewards_history = []
    terminated_history = []
    truncated_history = []
    
    print(f"\n开始执行动作（最多 {max_steps} 步）...\n")
    
    for step in range(max_steps):
        # 随机动作
        action = np.random.uniform(-1, 1, size=(1, cfg.action_dim))
        
        # 执行动作
        result = venv.step(action)
        
        # 解包结果
        if len(result) == 5:
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = result
        elif len(result) == 4:
            obs_venv, reward_venv, done_venv, info_venv = result
            terminated_venv = done_venv
            truncated_venv = np.zeros_like(done_venv)
        else:
            raise ValueError(f"Unexpected step result length: {len(result)}")
        
        # 记录
        reward = reward_venv[0] if isinstance(reward_venv, np.ndarray) else reward_venv
        terminated = terminated_venv[0] if isinstance(terminated_venv, np.ndarray) else terminated_venv
        truncated = truncated_venv[0] if isinstance(truncated_venv, np.ndarray) else truncated_venv
        
        rewards_history.append(reward)
        terminated_history.append(terminated)
        truncated_history.append(truncated)
        
        total_reward += reward
        step_count += 1
        
        # 每10步或有奖励/终止时打印
        if step % 10 == 0 or reward != 0 or terminated or truncated:
            print(f"Step {step:3d}: reward={reward:7.4f}, terminated={terminated}, truncated={truncated}, total_reward={total_reward:7.4f}")
        
        # 检查是否终止
        if terminated or truncated:
            print(f"\n{'='*80}")
            print(f"Episode 终止于第 {step_count} 步")
            print(f"  - 原因: {'terminated' if terminated else 'truncated'}")
            print(f"  - 总奖励: {total_reward:.4f}")
            print(f"{'='*80}")
            break
    else:
        print(f"\n{'='*80}")
        print(f"Episode 达到最大步数 {max_steps} 但未终止")
        print(f"  - 总奖励: {total_reward:.4f}")
        print(f"  - 最后一步 terminated: {terminated_history[-1]}")
        print(f"  - 最后一步 truncated: {truncated_history[-1]}")
        print(f"{'='*80}")
    
    # 统计分析
    print("\n" + "="*80)
    print("统计分析")
    print("="*80)
    
    rewards_array = np.array(rewards_history)
    terminated_array = np.array(terminated_history)
    truncated_array = np.array(truncated_history)
    
    print(f"\n奖励统计:")
    print(f"  - 总步数: {len(rewards_history)}")
    print(f"  - 总奖励: {total_reward:.4f}")
    print(f"  - 平均奖励: {np.mean(rewards_array):.4f}")
    print(f"  - 最大奖励: {np.max(rewards_array):.4f}")
    print(f"  - 最小奖励: {np.min(rewards_array):.4f}")
    print(f"  - 非零奖励步数: {np.count_nonzero(rewards_array)}")
    
    print(f"\n终止信号统计:")
    print(f"  - terminated 为 True 的次数: {np.sum(terminated_array)}")
    print(f"  - truncated 为 True 的次数: {np.sum(truncated_array)}")
    
    if np.sum(terminated_array) > 0:
        print(f"  - 首次 terminated 的步数: {np.argmax(terminated_array)}")
    if np.sum(truncated_array) > 0:
        print(f"  - 首次 truncated 的步数: {np.argmax(truncated_array)}")
    
    # 诊断建议
    print("\n" + "="*80)
    print("诊断结果")
    print("="*80)
    
    issues = []
    
    if total_reward == 0:
        issues.append("❌ 问题1: 总奖励为0")
        print("\n❌ 问题1: 总奖励为0")
        print("   可能原因:")
        print("   - 环境配置错误，奖励函数未启用")
        print("   - 随机动作无法完成任务（正常情况）")
        print("   - 奖励缩放配置问题")
    else:
        print(f"\n✓ 奖励正常: 总奖励 = {total_reward:.4f}")
    
    if np.sum(terminated_array) == 0 and np.sum(truncated_array) == 0:
        issues.append("❌ 问题2: Episode 从未终止")
        print("\n❌ 问题2: Episode 从未终止")
        print("   可能原因:")
        print("   - 环境的 max_episode_steps 设置过大")
        print("   - MultiStepWrapper 的 reset_within_step=True 可能阻止了终止信号")
        print("   - 环境本身不发送终止信号")
    else:
        print(f"\n✓ 终止信号正常")
    
    if len(issues) == 0:
        print("\n✅ 环境运行正常！")
    else:
        print(f"\n发现 {len(issues)} 个问题需要解决")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

