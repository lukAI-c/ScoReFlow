# MIT License
# Copyright (c) 2025 ReinFlow Authors - Distributional RL Extension

"""
PPO Buffer with Distributional Critic (C51) Support

继承自 PPOFlowBuffer，修改 get_value 的调用方式以支持分布式 Critic。
核心修改：使用 model.get_value(obs) 而非 critic.forward(obs)
"""

import numpy as np
import torch
import logging
log = logging.getLogger(__name__)

from agent.finetune.reinflow.buffer import PPOBuffer, PPOFlowBuffer


class PPOBufferDistribute(PPOBuffer):
    """
    支持分布式 Critic 的 PPO Buffer
    
    主要修改:
        - update() 接收整个 model 而非 critic
        - update_adv_returns() 使用 model.get_value() 获取价值
    """
    
    @torch.no_grad
    def update(self, obs_venv: dict, model, device='cpu'):
        '''
        obs_venv: dict containing numpy.ndarray
        model: 完整的 PPOFlow 模型 (支持分布式 Critic)
        '''
        # normalize reward with running variance
        # self.normalize_reward()
        self.update_adv_returns(obs_venv, model, device)
    
    @torch.no_grad
    def update_adv_returns(self, obs_venv, model, buffer_device='cpu'):
        '''
        使用 model.get_value() 获取价值，支持分布式 Critic
        
        Args:
            obs_venv: dict containing numpy.ndarray
            model: 完整的 PPOFlow 模型
            buffer_device: 'cpu' or 'cuda'
        '''
        # bootstrap value with GAE if not terminal
        obs_venv_ts = {
            "state": torch.from_numpy(obs_venv["state"])
            .float()
            .to(self.device)
        }
        
        self.advantages_trajs = (
            np.zeros((self.n_steps, self.n_envs)) 
            if buffer_device == 'cpu' 
            else torch.zeros(self.n_steps, self.n_envs, device=self.device)
        )
        
        lastgaelam = 0
        for t in reversed(range(self.n_steps)):
            # get V(s_t+1)
            if t == self.n_steps - 1:
                # 核心修改：使用 model.get_value() 支持分布式 Critic
                nextvalues = self._get_value_from_model(model, obs_venv_ts)
                nextvalues = nextvalues.reshape(1, -1)
                nextvalues = (
                    nextvalues.cpu().numpy() 
                    if buffer_device == 'cpu' 
                    else nextvalues.to(self.device)
                )
            else:
                nextvalues = self.value_trajs[t + 1]
            
            # delta = r + gamma*V(st+1) - V(st)
            non_terminal = 1.0 - self.terminated_trajs[t]
            delta = (
                self.reward_trajs[t] * self.reward_scale_const
                + self.gamma * nextvalues * non_terminal
                - self.value_trajs[t]
            )
            # A = delta_t + gamma*lambda*delta_{t+1} + ...
            self.advantages_trajs[t] = lastgaelam = (
                delta
                + self.gamma * self.gae_lambda * non_terminal * lastgaelam
            )
        self.returns_trajs = self.advantages_trajs + self.value_trajs
    
    def _get_value_from_model(self, model, obs):
        """
        从模型获取价值估计
        
        支持:
            1. model.get_value() - 分布式 Critic (C51)
            2. model.critic.forward() - 标量 Critic
        """
        if hasattr(model, 'get_value'):
            # 分布式 Critic: 返回期望值
            return model.get_value(obs)
        elif hasattr(model, 'critic'):
            # 标量 Critic
            return model.critic.forward(obs).view(-1)
        else:
            # 直接是 critic 模块
            return model.forward(obs).view(-1)


class PPOFlowBufferDistribute(PPOFlowBuffer):
    """
    支持分布式 Critic 的 PPOFlow Buffer
    
    继承自 PPOFlowBuffer，用于 Flow Matching 策略
    """
    
    @torch.no_grad
    def update(self, obs_venv: dict, model, device='cpu'):
        '''
        obs_venv: dict containing numpy.ndarray
        model: 完整的 PPOFlow 模型 (支持分布式 Critic)
        '''
        self.normalize_reward()
        self.update_adv_returns(obs_venv, model, device)
    
    @torch.no_grad
    def update_adv_returns(self, obs_venv, model, buffer_device='cpu'):
        '''
        使用 model.get_value() 获取价值，支持分布式 Critic
        '''
        obs_venv_ts = {
            "state": torch.from_numpy(obs_venv["state"])
            .float()
            .to(self.device)
        }
        
        self.advantages_trajs = (
            np.zeros((self.n_steps, self.n_envs)) 
            if buffer_device == 'cpu' 
            else torch.zeros(self.n_steps, self.n_envs, device=self.device)
        )
        
        lastgaelam = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                nextvalues = self._get_value_from_model(model, obs_venv_ts)
                nextvalues = nextvalues.reshape(1, -1)
                nextvalues = (
                    nextvalues.cpu().numpy()
                    if buffer_device == 'cpu'
                    else nextvalues.to(self.device)
                )
            else:
                nextvalues = self.value_trajs[t + 1]

            # delta = r + gamma*V(st+1) - V(st)
            non_terminal = 1.0 - self.terminated_trajs[t]
            delta = (
                self.reward_trajs[t] * self.reward_scale_const
                + self.gamma * nextvalues * non_terminal
                - self.value_trajs[t]
            )
            # A = delta_t + gamma*lambda*delta_{t+1} + ...
            self.advantages_trajs[t] = lastgaelam = (
                delta
                + self.gamma * self.gae_lambda * non_terminal * lastgaelam
            )
        self.returns_trajs = self.advantages_trajs + self.value_trajs

    def _get_value_from_model(self, model, obs):
        """从模型获取价值估计，支持分布式 Critic"""
        if hasattr(model, 'get_value'):
            return model.get_value(obs)
        elif hasattr(model, 'critic'):
            return model.critic.forward(obs).view(-1)
        else:
            return model.forward(obs).view(-1)
        
class PPOFlowImgBufferDistribute(PPOFlowBufferDistribute):
    """
    支持分布式 Critic 的 PPOFlow 图像 Buffer

    用于图像输入任务 (如 robomimic square)
    继承自 PPOFlowBufferDistribute，添加图像观测处理
    """

    def __init__(self,
                 n_steps,
                 n_envs,
                 n_ft_denoising_steps,
                 horizon_steps,
                 act_steps,
                 action_dim,
                 n_cond_step,
                 obs_dim,  # dict: {'rgb': (C, H, W), 'state': (D,)}
                 save_full_observation,
                 furniture_sparse_reward,
                 best_reward_threshold_for_success,
                 reward_scale_running,
                 gamma,
                 gae_lambda,
                 reward_scale_const,
                 aug,  # image augmentation
                 fix_nextvalue_augment_bug: bool,
                 device,
                 log_prob_cfg_dict: dict
                 ):
        # 注意：obs_dim 是 dict，不是 int
        # 调用父类时传入 obs_dim=obs_dim['state'][0] 会出错，所以我们自己处理
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.ft_denoising_steps = n_ft_denoising_steps
        self.horizon_steps = horizon_steps
        self.act_steps = act_steps
        self.action_dim = action_dim
        self.n_cond_step = n_cond_step
        self.obs_dim = obs_dim  # dict
        self.save_full_observation = save_full_observation
        self.furniture_sparse_reward = furniture_sparse_reward
        self.best_reward_threshold_for_success = best_reward_threshold_for_success
        self.reward_scale_running = reward_scale_running
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reward_scale_const = reward_scale_const
        self.device = device

        self.aug = aug
        self.log_prob_cfg_dict = log_prob_cfg_dict
        self.fix_nextvalue_augment_bug = fix_nextvalue_augment_bug

        # Reward normalization
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 1e-4

        # Episode tracking
        self.episode_reward = np.zeros(n_envs)
        self.episode_reward_list = []
        self.episode_best_reward = -np.inf * np.ones(n_envs)
        self.episode_best_reward_list = []
        self.success_rate = 0.0
        self.avg_episode_reward = 0.0
        self.avg_best_reward = 0.0
        self.num_episodes = 0

    def reset(self):
        """重置 buffer，使用 GPU tensor"""
        # 图像观测
        self.obs_trajs = {
            k: torch.zeros(
                (self.n_steps, self.n_envs, self.n_cond_step, *self.obs_dim[k]),
                dtype=torch.float32, device=self.device
            )
            for k in self.obs_dim
        }
        # 动作链
        self.chains_trajs = torch.zeros(
            (self.n_steps, self.n_envs, self.ft_denoising_steps + 1, self.horizon_steps, self.action_dim),
            dtype=torch.float32, device=self.device
        )
        self.reward_trajs = torch.zeros((self.n_steps, self.n_envs), dtype=torch.float32, device=self.device)
        self.terminated_trajs = torch.zeros((self.n_steps, self.n_envs), dtype=torch.float32, device=self.device)
        self.firsts_trajs = torch.zeros((self.n_steps + 1, self.n_envs), dtype=torch.float32, device=self.device)
        self.value_trajs = torch.zeros((self.n_steps, self.n_envs), dtype=torch.float32, device=self.device)
        self.logprobs_trajs = torch.zeros((self.n_steps, self.n_envs), dtype=torch.float32, device=self.device)

    def add(self, step, prev_obs_venv, chains_actions_venv, reward_venv, terminated_venv, truncated_venv):
        """添加一步数据"""
        # 图像观测
        for k in self.obs_trajs:
            self.obs_trajs[k][step] = torch.from_numpy(prev_obs_venv[k]).float().to(self.device)

        self.chains_trajs[step] = chains_actions_venv
        self.reward_trajs[step] = torch.from_numpy(reward_venv).float().to(self.device)
        self.terminated_trajs[step] = torch.from_numpy(terminated_venv).float().to(self.device)
        self.firsts_trajs[step + 1] = torch.from_numpy(terminated_venv | truncated_venv).float().to(self.device)

        # Episode reward tracking
        self.episode_reward += reward_venv
        done_venv = terminated_venv | truncated_venv
        for i, done in enumerate(done_venv):
            if done:
                self.episode_reward_list.append(self.episode_reward[i])
                self.episode_best_reward_list.append(self.episode_best_reward[i])
                self.episode_reward[i] = 0.0
                self.episode_best_reward[i] = -np.inf

    def normalize_reward(self):
        """Reward normalization using running statistics"""
        if self.reward_scale_running:
            reward_np = self.reward_trajs.cpu().numpy()
            batch_mean = np.mean(reward_np)
            batch_var = np.var(reward_np)
            batch_count = reward_np.size

            delta = batch_mean - self.reward_mean
            total_count = self.reward_count + batch_count
            self.reward_mean += delta * batch_count / total_count
            m_a = self.reward_var * self.reward_count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta ** 2 * self.reward_count * batch_count / total_count
            self.reward_var = M2 / total_count
            self.reward_count = total_count

            self.reward_trajs = self.reward_trajs / (torch.sqrt(torch.tensor(self.reward_var, device=self.device)) + 1e-8)
    @torch.no_grad()
    def update_img(self, obs_venv: dict, model):
        """
        更新 buffer，支持分布式 Critic

        Args:
            obs_venv: 最后一步的观测
            model: 完整的 PPOFlow 模型
        """
        self.normalize_reward()
        self.update_value_logprob(model)
        self.update_adv_returns_img(obs_venv, model)

    @torch.no_grad()
    def update_value_logprob(self, model):
        """
        计算 value 和 logprob（带图像增强）

        方案2：将增强后的观测保存回 self.obs_trajs，确保：
        - oldlogprobs 使用增强后的观测计算
        - make_dataset() 返回增强后的观测
        - newlogprobs 在 model.loss() 中也使用相同的增强后观测
        这样 ratio 在 update_epoch=0 时等于 1.0
        """
        # 图像增强：对整个 trajectory 做增强，然后保存回 self.obs_trajs
        if self.aug:
            # (n_steps, n_envs, n_cond_step, C, H, W) -> (n_steps*n_envs*n_cond_step, C, H, W)
            rgb = self.obs_trajs["rgb"].flatten(0, 2)
            rgb = self.aug(rgb)
            # 将增强后的观测保存回 self.obs_trajs，这样 make_dataset() 也会使用增强后的数据
            self.obs_trajs["rgb"] = rgb.reshape(
                self.n_steps, self.n_envs, self.n_cond_step, *self.obs_dim['rgb']
            )

        # 逐步计算 value 和 logprob
        for step in range(self.n_steps):
            # 注意：ViTCritic 期望格式 (B, T_rgb, C, H, W)
            # 不做 flatten，保持 (e, t, C, H, W) 格式
            obs_step = {
                key: self.obs_trajs[key][step].to(self.device)  # (e, t, ...)
                for key in self.obs_dim
            }

            # 使用 model.get_value() 支持分布式 Critic
            value = self._get_value_from_model(model, obs_step)
            self.value_trajs[step] = value.view(self.n_envs)

            # 计算 logprob
            chains = self.chains_trajs[step]  # (e, T+1, h, a)
            logprob = model.get_logprobs(
                cond=obs_step,
                x_chain=chains,
                get_entropy=False,
                get_chains_stds=False,
                **self.log_prob_cfg_dict
            )
            self.logprobs_trajs[step] = logprob.view(self.n_envs)

    @torch.no_grad()
    def update_adv_returns_img(self, obs_venv, model):
        """计算 advantage 和 returns（图像版本）"""
        # 处理最后一步的观测
        obs_last = {
            key: torch.from_numpy(obs_venv[key]).float().to(self.device)
            for key in self.obs_dim
        }

        # 图像增强（如果需要且修复 bug）
        if self.aug and self.fix_nextvalue_augment_bug:
            rgb = obs_last["rgb"].flatten(0, 1)  # (e*t, C, H, W)
            rgb = self.aug(rgb)
            obs_last["rgb"] = rgb.reshape(self.n_envs, self.n_cond_step, *self.obs_dim['rgb'])

        # obs_last_flat = {
        #     key: obs_last[key].flatten(0, 1)
        #     for key in self.obs_dim
        # }

        self.advantages_trajs = torch.zeros(
            (self.n_steps, self.n_envs), dtype=torch.float32, device=self.device
        )

        lastgaelam = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                # 使用 model.get_value() 支持分布式 Critic
                nextvalues = self._get_value_from_model(model, obs_last)
                nextvalues = nextvalues.view(1, -1)
            else:
                nextvalues = self.value_trajs[t + 1].unsqueeze(0)

            non_terminal = 1.0 - self.terminated_trajs[t]
            delta = (
                self.reward_trajs[t] * self.reward_scale_const
                + self.gamma * nextvalues.squeeze() * non_terminal
                - self.value_trajs[t]
            )
            self.advantages_trajs[t] = lastgaelam = (
                delta + self.gamma * self.gae_lambda * non_terminal * lastgaelam
            )

        self.returns_trajs = self.advantages_trajs + self.value_trajs
        # 调试：打印 reward 和 returns 的范围
        reward_flat = self.reward_trajs.flatten()
        value_flat = self.value_trajs.flatten()
        adv_flat = self.advantages_trajs.flatten()
        returns_flat = self.returns_trajs.flatten()
        log.info(f"[Buffer Debug] reward_range=[{reward_flat.min():.2f}, {reward_flat.max():.2f}], "
                f"value_range=[{value_flat.min():.2f}, {value_flat.max():.2f}], "
                f"adv_range=[{adv_flat.min():.2f}, {adv_flat.max():.2f}], "
                f"returns_range=[{returns_flat.min():.2f}, {returns_flat.max():.2f}]")

    def make_dataset(self):
        """生成训练数据集"""
        obs = {
            "state": self.obs_trajs["state"].clone().detach().flatten(0, 1),
            "rgb": self.obs_trajs["rgb"].clone().detach().flatten(0, 1)
        }
        chains = self.chains_trajs.flatten(0, 1)
        returns = self.returns_trajs.flatten(0, 1)
        values = self.value_trajs.flatten(0, 1)
        advantages = self.advantages_trajs.flatten(0, 1)
        logprobs = self.logprobs_trajs.flatten(0, 1)

        return obs, chains, returns, values, advantages, logprobs

    @torch.no_grad()
    def summarize_episode_reward(self):
        """统计 episode 奖励 (使用 firsts_trajs 计算完整 episode)"""
        episodes_start_end = []
        # Convert firsts_trajs to numpy for processing
        firsts_trajs_np = self.firsts_trajs.cpu().numpy()

        for env_ind in range(self.n_envs):
            env_steps = np.where(firsts_trajs_np[:, env_ind] == 1)[0]
            for i in range(len(env_steps) - 1):
                start = env_steps[i]
                end = env_steps[i + 1]
                if end - start > 1:
                    episodes_start_end.append((env_ind, start, end - 1))

        if len(episodes_start_end) > 0:
            # Select reward_trajs using numpy slicing
            reward_trajs_split = [
                self.reward_trajs[start:end + 1, env_ind].cpu().numpy()
                for env_ind, start, end in episodes_start_end
            ]
            self.num_episode_finished = len(reward_trajs_split)

            # Calculating episode_reward using numpy
            episode_reward = np.array([np.sum(reward_traj) for reward_traj in reward_trajs_split])

            if self.furniture_sparse_reward:
                episode_best_reward = episode_reward
            else:
                episode_best_reward = np.array([
                    np.max(reward_traj) / self.act_steps
                    for reward_traj in reward_trajs_split
                ])

            # Compute metrics
            self.avg_episode_reward = np.mean(episode_reward)
            self.avg_best_reward = np.mean(episode_best_reward)
            self.success_rate = np.mean(
                episode_best_reward >= self.best_reward_threshold_for_success
            )

            # Calculate standard deviations
            self.std_episode_reward = np.std(episode_reward)
            self.std_best_reward = np.std(episode_best_reward)
            self.std_success_rate = np.std(
                episode_best_reward >= self.best_reward_threshold_for_success
            )

            # Calculate average length of valid episodes and its standard deviation
            episode_lengths = np.array([end - start + 1 for _, start, end in episodes_start_end]) * self.act_steps
            self.avg_episode_length = np.mean(episode_lengths)
            self.std_episode_length = np.std(episode_lengths)
        else:
            self.num_episode_finished = 0
            self.avg_episode_reward = 0
            self.avg_best_reward = 0
            self.success_rate = 0
            self.std_episode_reward = 0
            self.std_best_reward = 0
            self.std_success_rate = 0
            self.avg_episode_length = 0
            self.std_episode_length = 0

    def update_full_obs(self):
        """更新完整观测"""
        pass

    def save_full_obs(self, info_venv):
        """保存完整观测"""
        pass

    def get_explained_var(self, values, returns):
        """计算 explained variance"""
        var_returns = torch.var(returns)
        if var_returns == 0:
            return 0.0
        return 1 - torch.var(returns - values) / var_returns
