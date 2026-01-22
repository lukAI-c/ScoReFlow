# MIT License

# Copyright (c) 2025 ReinFlow Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
Parent PPO fine-tuning agent class.
"""
from typing import Optional
import torch
import logging
from util.scheduler import CosineAnnealingWarmupRestarts, WarmupReduceLROnPlateau
from util.scheduler_simple import CustomScheduler
log = logging.getLogger(__name__)
from agent.finetune.reinflow.train_agent import TrainAgent
from util.reward_scaling import RunningRewardScaler
from util.logging_custom import create_bordered_text
########################################################################
# appended by ReinFlow Authors
import os
import numpy as np
from util.timer import Timer
import os
import numpy as np
import pickle
import wandb
import pandas as pd
from util.reproducibility import set_seed_everywhere
########################################################################
class TrainPPOAgent(TrainAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.resume = cfg.get('resume_path', False)
        
        # Batch size for logprobs calculations after an iteration --- 
        # prevent out of memory if using a single batch
        self.logprob_batch_size = cfg.train.get("logprob_batch_size", 10000)
        assert (
            self.logprob_batch_size % self.n_envs == 0
        ), "logprob_batch_size must be divisible by n_envs"

        # note the discount factor gamma here is applied to reward every act_steps, instead of every env step
        self.gamma = cfg.train.gamma

        # Warm up period for critic before actor updates
        self.n_critic_warmup_itr = cfg.train.n_critic_warmup_itr
        actor_params = self.model.get_actor_params() if hasattr(self.model, 'get_actor_params') else self.model.actor_ft.parameters()

        self.actor_optimizer = torch.optim.AdamW(
            # self.model.actor_ft.parameters(),
            actor_params,
            lr=cfg.train.actor_lr,
            weight_decay=cfg.train.actor_weight_decay,
        )
        self.actor_lr_type = cfg.train.actor_lr_scheduler.get("type", "cosine")
        if self.actor_lr_type == "cosine":
            self.actor_lr_scheduler = CosineAnnealingWarmupRestarts(
                self.actor_optimizer,
                first_cycle_steps=cfg.train.actor_lr_scheduler.first_cycle_steps,
                cycle_mult=1.0,
                max_lr=cfg.train.actor_lr,
                min_lr=cfg.train.actor_lr_scheduler.min_lr,
                warmup_steps=cfg.train.actor_lr_scheduler.warmup_steps,
                gamma=1.0,
            )
        elif self.actor_lr_type == "plateau":
            self.actor_lr_scheduler = WarmupReduceLROnPlateau(
                self.actor_optimizer,
                warmup_steps = cfg.train.actor_lr_scheduler.warmup_steps,
                target_lr = cfg.train.actor_lr,
                mode='max',         # Use 'max' for rewards if they are your metric
                min_lr = cfg.train.actor_lr_scheduler.min_lr,
                factor=0.6, 
                patience=4, 
                threshold=20,
                verbose=True
            )
        elif self.actor_lr_type=='constant_warmup':
            self.actor_lr_scheduler=CustomScheduler(self.actor_optimizer,
                                                    'constant_warmup', 
                                                  min=cfg.train.actor_lr_scheduler.min_lr, 
                                                  warmup_steps=cfg.train.actor_lr_scheduler.warmup_steps,
                                                  max=cfg.train.actor_lr)
        elif self.actor_lr_type=='cosine_custom':
            self.actor_lr_scheduler=CustomScheduler(self.actor_optimizer,
                                                     schedule_type='cosine', 
                                                     max=cfg.train.actor_lr,
                                                     hold_steps=cfg.train.actor_lr_scheduler.hold_steps,
                                                     anneal_steps=cfg.train.actor_lr_scheduler.anneal_steps,
                                                     min=cfg.train.actor_lr_scheduler.min_lr)
        else:
            raise ValueError(f"Invalid actor_lr_type: {self.actor_lr_type}")
        
        self.critic_optimizer = torch.optim.AdamW(
            self.model.critic.parameters(),
            lr=cfg.train.critic_lr,
            weight_decay=cfg.train.critic_weight_decay,
        )
        self.critic_lr_type = cfg.train.critic_lr_scheduler.get("type", "cosine")
        if self.critic_lr_type == "cosine":
            self.critic_lr_scheduler = CosineAnnealingWarmupRestarts(
                self.critic_optimizer,
                first_cycle_steps=cfg.train.critic_lr_scheduler.first_cycle_steps,
                cycle_mult=1.0,
                max_lr=cfg.train.critic_lr,
                min_lr=cfg.train.critic_lr_scheduler.min_lr,
                warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
                gamma=1.0,
            )
        elif self.critic_lr_type == "plateau":
            self.critic_lr_scheduler = WarmupReduceLROnPlateau(
                self.critic_optimizer,
                warmup_steps = cfg.train.critic_lr_scheduler.warmup_steps,
                target_lr = cfg.train.critic_lr,
                mode='max',         # Use 'max' for rewards if they are your metric
                min_lr = cfg.train.critic_lr_scheduler.min_lr,
                factor=0.6, 
                patience=4, 
                threshold=20,
                verbose=True
            )
        elif self.critic_lr_type=='constant_warmup':
            self.critic_lr_scheduler=CustomScheduler(self.critic_optimizer,
                                                     'constant_warmup', 
                                                        min=cfg.train.critic_lr_scheduler.min_lr, 
                                                        warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
                                                        max=cfg.train.critic_lr)
        elif self.critic_lr_type=='cosine_custom':
            self.critic_lr_scheduler=CustomScheduler(self.critic_optimizer,
                                                     schedule_type='cosine', 
                                                     max=cfg.train.critic_lr,
                                                     hold_steps=cfg.train.critic_lr_scheduler.hold_steps,
                                                     anneal_steps=cfg.train.critic_lr_scheduler.anneal_steps,
                                                     min=cfg.train.critic_lr_scheduler.min_lr)
        else:
            raise ValueError(f"Invalid critic_lr_type: {self.critic_lr_type}")
        
        self.visualize_lr(cfg)
        
        # Generalized advantage estimation
        self.gae_lambda: float = cfg.train.get("gae_lambda", 0.95)

        # If specified, stop gradient update once KL difference reaches it
        self.target_kl: Optional[float] = cfg.train.target_kl

        # Number of times the collected data is used in gradient update
        self.update_epochs: int = cfg.train.update_epochs

        # Entropy loss coefficient
        self.ent_coef = cfg.train.get("ent_coef", 0.01)
        self.ent_coef_schedule_on = cfg.train.get("ent_coef_schedule_on", False)
        
        if self.ent_coef_schedule_on == True:
            # Entropy coefficient schedule parameters (方案一：带缓冲的线性衰减)
            self.ent_coef_schedule = cfg.train.get("ent_coef_schedule", "linear_decay")  # 'constant' or 'linear_decay'
            self.ent_coef_start = cfg.train.get("ent_coef_start", self.ent_coef)  # 初始熵系数
            self.ent_coef_end = cfg.train.get("ent_coef_end", 0.001)  # 最终熵系数
            self.ent_decay_start_itr = cfg.train.get("ent_decay_start_itr", 30)  # 开始衰减的 iteration
            self.ent_decay_end_itr = cfg.train.get("ent_decay_end_itr", int(cfg.train.n_train_itr * 0.8))  # 结束衰减的 iteration
        else: 
            self.ent_coef_schedule = "constant"

        # Value loss coefficient
        self.vf_coef = cfg.train.get("vf_coef", 0.5)

        # Whether to use running reward scaling
        self.reward_scale_running: bool = cfg.train.reward_scale_running
        
        # Scaling reward with constant
        self.reward_scale_const: float = cfg.train.get("reward_scale_const", 1)
        
        if self.reward_scale_running:
            self.running_reward_scaler = RunningRewardScaler(cfg.env.n_envs)

        # appended by ReinFlow Authors: use base policy to regularize policy. 
        self.use_bc_loss: bool = cfg.train.get("use_bc_loss", False)
        self.bc_loss_type: bool = cfg.train.get("bc_loss_type", None) if self.use_bc_loss else None
        bc_loss_type_candidates=['W2', 'KL', None]
        if not self.bc_loss_type in bc_loss_type_candidates:
            raise ValueError(f"Unsupported bc_loss_type={self.bc_loss_type}. It must be in the following: {bc_loss_type_candidates}")
        self.bc_coeff: float = cfg.train.get("bc_loss_coeff", 0.0)
        
        # appended by ReinFlow Authors:
        self.current_best_reward = np.float32('-inf')
        self.is_best_so_far = False 
        self.total_steps = self.n_steps * self.n_envs # total number of actions
        self.buffer = None
        self.verbose = cfg.train.get('verbose', False)
        self.actor_update_freq = cfg.train.get('actor_update_freq',1)
        self.actor_update_epoch = cfg.train.get('actor_update_epoch',1)
        self.denoising_steps = cfg.get('denoising_steps', 1)  # 1 is for gaussian policy
        self.skip_initial_eval = False 
    def update_ent_coef_schedule(self):
        """
        方案一：带缓冲的线性衰减 (Linear Decay with Warmup & Cooldown)

        训练分为三个阶段：
        1. 探索期 (Warmup): 保持高熵，让模型先"把路走通"
        2. 衰减期 (Decay): 线性降低熵权，引导模型收束方差
        3. 精调期 (Cooldown): 保持低熵（底噪），专注刷分，死磕精度

        Config 参数:
            ent_coef_schedule: 'constant' 或 'linear_decay'
            ent_coef_start: 初始熵系数 (默认 0.02)
            ent_coef_end: 最终熵系数 (默认 0.001)
            ent_decay_start_itr: 开始衰减的 iteration (默认 50)
            ent_decay_end_itr: 结束衰减的 iteration (默认 n_train_itr * 0.8)
        """
        if self.ent_coef_schedule == 'constant':
            # 保持不变
            return

        elif self.ent_coef_schedule == 'linear_decay':
            # 1. 还没有到衰减时间，保持高熵
            if self.itr < self.ent_decay_start_itr:
                self.ent_coef = self.ent_coef_start

            # 2. 已经过了衰减时间，保持低熵
            elif self.itr >= self.ent_decay_end_itr:
                self.ent_coef = self.ent_coef_end

            # 3. 正在衰减中 (Linear Interpolation)
            else:
                # 计算进度 0.0 -> 1.0
                progress = (self.itr - self.ent_decay_start_itr) / (self.ent_decay_end_itr - self.ent_decay_start_itr)
                # 公式: Current = Start - Progress * (Start - End)
                self.ent_coef = self.ent_coef_start - progress * (self.ent_coef_start - self.ent_coef_end)
        else:
            raise ValueError(f"Unsupported ent_coef_schedule: {self.ent_coef_schedule}. Use 'constant' or 'linear_decay'.")
        
    def visualize_lr(self, cfg):
        steps = []
        critic_lrs = []
        actor_lrs = []
        for step in range(cfg.train.n_train_itr):
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            steps.append(step)
            critic_lrs.append(self.critic_optimizer.param_groups[0]["lr"])
            actor_lrs.append(self.actor_optimizer.param_groups[0]["lr"])
        import matplotlib.pyplot as plt
        plt.subplot(1,2,1)
        plt.plot(steps, actor_lrs, label='actor', color = 'blue')
        plt.legend(loc='upper right')
        plt.subplot(1,2,2)
        plt.plot(steps, critic_lrs,label='critic', color='red')
        plt.legend(loc='upper right')
        lr_save_path = os.path.join(self.logdir, 'test_lr_schedulers.png')
        plt.savefig(lr_save_path)
        log.info(f"learning rate saved to {lr_save_path}")
        plt.close()
        
        if isinstance(self.actor_lr_scheduler, CustomScheduler):
            self.actor_lr_scheduler.reset()
        if isinstance(self.critic_lr_scheduler, CustomScheduler):
            self.critic_lr_scheduler.reset()
        
        self.print_architecture()
    
    def print_architecture(self):
        import os
        arc_path=os.path.join(self.logdir, 'architecture.log')
        with open(arc_path, mode='w') as arc_file:
            arc_file.write(f"self.model=\n{self.model}\nnumber of parameters: {sum([p.numel() for p in self.model.parameters()])/1e6:.2f} M")
        log.info(f"architecture wrote to file {arc_path}")
        arc_file.close()
        
    def reset_actor_optimizer(self):
        """Not used anywhere currently"""
        actor_params = self.model.get_actor_params() if hasattr(self.model, 'get_actor_params') else self.model.actor_ft.parameters()
        new_optimizer = torch.optim.AdamW(
            # self.model.actor_ft.parameters(),
            actor_params,
            lr=self.cfg.train.actor_lr,
            weight_decay=self.cfg.train.actor_weight_decay,
        )
        new_optimizer.load_state_dict(self.actor_optimizer.state_dict())
        self.actor_optimizer = new_optimizer
        
        new_scheduler = CosineAnnealingWarmupRestarts(
            self.actor_optimizer,
            first_cycle_steps=self.cfg.train.actor_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=self.cfg.train.actor_lr,
            min_lr=self.cfg.train.actor_lr_scheduler.min_lr,
            warmup_steps=self.cfg.train.actor_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        new_scheduler.load_state_dict(self.actor_lr_scheduler.state_dict())
        self.actor_lr_scheduler = new_scheduler
        log.info("Reset actor optimizer")
    
    def update_lr(self):
        self.critic_lr_scheduler.step()
        if self.itr >= self.n_critic_warmup_itr: 
            self.actor_lr_scheduler.step()
        log.info(f"""learning rate updated. actor_lr={self.actor_optimizer.param_groups[0]["lr"]:.2e}, critic_lr={self.critic_optimizer.param_groups[0]["lr"]:.2e}""")
    
    ########################################################################
    # appended by ReinFlow Authors
    def prepare_video_path(self):
        # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
        self.options_venv = [{} for _ in range(self.n_envs)]
        log.info(f"[Video Debug] itr={self.itr}, render_freq={self.render_freq}, render_video={self.render_video}, n_render={self.n_render}")
        if self.itr % self.render_freq == 0 and self.render_video:
            for env_ind in range(self.n_render):
                # self.options_venv[env_ind]["video_path"] = os.path.join(
                #     self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                # )
                video_path = os.path.join(self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4")
                self.options_venv[env_ind]["video_path"] = video_path
                log.info(f"[Video Debug] Will save video to: {video_path}")

    def set_model_mode(self):
        # Define train or eval - all envs restart
        if self.skip_initial_eval and self.itr ==0:
            self.eval_mode = False 
        else:
            if self.resume:
                self.eval_mode = True
                self.resume = False
            else:
                self.eval_mode = self.itr % self.val_freq == 0 and not self.force_train
        self.model.eval() if self.eval_mode else self.model.train()
        self.last_itr_eval = self.eval_mode

    def prepare_run(self):
        # Start training loop
        self.timer = Timer()
        self.run_results = []
        self.cnt_train_step = 0
        self.last_itr_eval = False
        self.done_venv = np.zeros((1,self.n_envs))
        # 创建专门的训练结果保存文件夹
        self.results_dir = os.path.join(self.logdir, 'training_results')
        os.makedirs(self.results_dir, exist_ok=True)
        # 初始化训练日志文件
        self.metrics_log_path = os.path.join(self.results_dir, 'training_metrics.txt')
        self.metrics_csv_path = os.path.join(self.results_dir, 'training_metrics.csv')
        # 写入 CSV 表头
        with open(self.metrics_csv_path, 'w') as f:
            f.write("itr,step,mode,success_rate,avg_episode_reward,avg_best_reward,avg_episode_length,actor_lr,critic_lr,loss,pg_loss,value_loss,entropy_loss,approx_kl,clip_frac\n")
        log.info(f"Training results folder: {self.results_dir}")
        log.info(f"Training metrics will be saved to:\n  {self.metrics_log_path}\n  {self.metrics_csv_path}")
    
    def reset_env(self, buffer_device='cpu'):
        # set_seed_everywhere(self.seed)
        # Reset env before iteration starts (1) if specified, (2) at eval mode, or (3) right after eval mode
        if self.reset_at_iteration or self.eval_mode or self.last_itr_eval:
            self.prev_obs_venv = self.reset_env_all(options_venv=self.options_venv)
            self.buffer.firsts_trajs[0] = 1
        else:
            # if done at the end of last iteration, the envs are just reset
            if buffer_device == 'cpu':
                self.buffer.firsts_trajs[0] = self.done_venv
            else:
                self.buffer.firsts_trajs[0] = torch.from_numpy(self.done_venv).float().to(buffer_device)
    
    def save_model(self):
        """
        overload. 
        saves model to disk; no ema recorded. 
        TODO: save ema
        """
        data = {
            "itr": self.itr,
            "model": self.model.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_lr_scheduler": self.actor_lr_scheduler.state_dict(),
            "critic_lr_scheduler": self.critic_lr_scheduler.state_dict(),
        }
        
        # always save the last model for resume of training. 
        save_path = os.path.join(self.checkpoint_dir,f"last.pt")
        torch.save(data, os.path.join(self.checkpoint_dir, save_path))
        # log.info(f"\n Saved last model at itr {self.itr} to {save_path}\n ")
        
        # optionally save intermediate models
        if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
            save_path = os.path.join(self.checkpoint_dir, f"state_{self.itr}.pt")
            torch.save(data, os.path.join(self.checkpoint_dir, save_path))
            log.info(f"\n Saved model at itr={self.itr} to {save_path}\n ")
        
        # save the best model evaluated so far 
        if self.is_best_so_far:
            save_path = os.path.join(self.checkpoint_dir,f"best.pt")
            torch.save(data, os.path.join(self.checkpoint_dir, save_path))
            log.info(f"\n Saved model with the highest evaluated average episode reward {self.current_best_reward:4.3f} to \n{save_path}\n ")
            self.is_best_so_far =False
    
    def plot_state_trajecories(self): 
        if not self.traj_plotter:
            return 
        if self.itr % self.render_freq == 0 and self.n_render > 0:
                self.traj_plotter(
                    obs_full_trajs=self.obs_full_trajs,
                    n_render=self.n_render,
                    max_episode_steps=self.max_episode_steps,
                    render_dir=self.render_dir,
                    itr=self.itr,
                )
    def resume_training(self):
        log.info(f"Resuming training...")
        data = torch.load(self.resume_path, weights_only=True, map_location=self.device)
        print(f"Recover checkpoint: data={data.keys()}")
        # recover itr, scheduler, sample number
        self.itr = data["itr"]
        self.cnt_train_step = self.itr * self.n_envs * self.act_steps * self.n_steps if 'cnt_train_steps' not in data.keys() else data["cnt_train_steps"]
        self.n_train_itr += self.itr # train for another xx iters. 
        log.info(f"Resume training from itr={self.itr}, total train steps={self.cnt_train_step}.")
        
        # load models
        if "model" in data.keys():
            self.model.load_state_dict(data["model"], strict=True)
            log.info(f"""Loaded full model, including {data["model"].keys()}""")
        elif "policy" in data.keys():
            self.model.actor_ft.policy.load_state_dict(data["policy"], strict=True)
            log.info(f"Loaded policy. Initialize critic and add noise network from scratch.")
        else:
            raise ValueError(f"Your saved checkpoint does not contain keys like 'model' and 'policy'. Please check what was wrong with your model saving functions. ")
        
        log.info(f"Successfully loaded model from path={self.resume_path}")
        
        # load optimizers
        self.actor_optimizer.load_state_dict(data["actor_optimizer"])
        self.critic_optimizer.load_state_dict(data["critic_optimizer"])
        log.info(f"Successfully loaded optimizers from path={self.resume_path}") 
        
        # load scheduler
        if 'actor_lr_scheduler' in data.keys():
            self.actor_lr_scheduler.load_state_dict(data["actor_lr_scheduler"])
            self.critic_lr_scheduler.load_state_dict(data["critic_lr_scheduler"])
            log.info(f"Successfully loaded schedulers from path={self.resume_path}")
        else:
            for _ in range(self.itr): # recover lr schedulers 
                self.actor_lr_scheduler.step()
                self.critic_lr_scheduler.step()
            log.info(f"No schedulers found in path={self.resume_path}. Automatically calibrate the newly initialized scheduler.")
    
    def update_step(self, batch):
        raise NotImplementedError
    
    def agent_update(self):
        # put all the samples in n_steps x n_envs in a line
        obs, samples, returns, values, advantages, logprobs = self.buffer.make_dataset()
        
        # Explained variation of future rewards using value function
        explained_var = self.buffer.get_explained_var(values, returns)
        
        clipfrac_list = []
        
        # generate a random minibatch of data. 
        for update_epoch in range(self.update_epochs):
            kl_change_too_much = False
            indices = torch.randperm(self.total_steps, device=self.device)
            for start in range(0, self.total_steps, self.batch_size):
                end = start + self.batch_size
                minibatch_idx = indices[start:end]

                batch = (
                    {"state": obs[minibatch_idx]},
                    samples[minibatch_idx],
                    returns[minibatch_idx],
                    values[minibatch_idx],
                    advantages[minibatch_idx],
                    logprobs[minibatch_idx]
                )

                # minibatch gradient descent
                pg_loss, entropy_loss, v_loss, clipfrac, approx_kl, ratio, bc_loss, std = self.update_step(batch) 
                
                loss = pg_loss + entropy_loss * self.ent_coef + v_loss * self.vf_coef + bc_loss * self.bc_coeff
                
                clipfrac_list += [clipfrac]
                
                # update policy and critic
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                
                loss.backward()
                
                if self.itr >= self.n_critic_warmup_itr:
                    if self.max_grad_norm : torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()
                self.critic_optimizer.step()

                # Stop gradient update if KL difference reaches target
                if self.target_kl and approx_kl > self.target_kl:
                    kl_change_too_much = True
                    break
            if kl_change_too_much:
                break
        clip_frac = np.mean(clipfrac_list)
        self.train_ret_dict = {
                "loss": loss,
                "pg loss": pg_loss,
                "value loss": v_loss,
                "entropy_loss": entropy_loss,
                "std": std,
                "approx kl": approx_kl,
                "ratio": ratio,
                "clip_frac": clip_frac,
                "explained variance": explained_var,
            }

    def log(self, train_prt_str_additional="", train_log_dict_additional={}):
        '''
        train_prt_str_additional: str, additional information in training that will be printed to log console that is not included in train_prt_str_basic
        train_log_dict_additional: dict, additional information in training that will be logged to wandb that is not included in train_log_dict_basic
        '''
        BOLDSTART = '\033[1m'
        BOLDEND = '\033[0m'

        self.run_results.append(
            {
                "itr": self.itr,
                "step": self.cnt_train_step,
            }
        )
        if self.save_trajs:
            self.run_results[-1]["self.obs_full_trajs"] = self.obs_full_trajs
            self.run_results[-1]["self.obs_trajs"] = self.obs_trajs
            self.run_results[-1]["action_trajs"] = self.samples_trajs
            self.run_results[-1]["self.reward_trajs"] = self.reward_trajs
        if self.itr % self.log_freq == 0:
            time = self.timer()
            self.run_results[-1]["time"] = time
            if self.eval_mode:
                # Updated evaluation log with avg ± std formatting
                log.info(create_bordered_text(
                    f"{BOLDSTART}Evaluation at itr {self.itr}{BOLDEND}:\n"
                    f"Model: {self.model.__class__.__name__}\n"
                    f"Environment: {self.env_name} x {self.n_envs}\n"
                    f"Num denoising steps: {self.denoising_steps}\n"
                    f"Seed: {self.seed}\n"
                    f"Success Rate: {self.buffer.success_rate * 100:3.2f}% ± {self.buffer.std_success_rate * 100:3.2f}%\n"
                    f"Episode Reward: {self.buffer.avg_episode_reward:8.2f} ± {self.buffer.std_episode_reward:8.2f}\n"
                    f"Best Reward (per action): {self.buffer.avg_best_reward:8.2f} ± {self.buffer.std_best_reward:8.2f}\n"
                    f"Episode Length: {self.buffer.avg_episode_length:8.2f} ± {self.buffer.std_episode_length:8.2f}\n"
                    f"Actor lr :{self.actor_optimizer.param_groups[0]['lr']:.2e}\n"
                    f"Critic lr: {self.critic_optimizer.param_groups[0]['lr']:.2e}"
                ))
                eval_dict={
                            "eval/success rate": self.buffer.success_rate,
                            "eval/avg episode reward": self.buffer.avg_episode_reward,
                            "eval/avg best reward": self.buffer.avg_best_reward,
                            "eval/avg episode length": self.buffer.avg_episode_length,
                            "eval/num episode": self.buffer.num_episode_finished,
                            "eval/std success rate": self.buffer.std_success_rate,
                            "eval/std episode reward": self.buffer.std_episode_reward,
                            "eval/std best reward": self.buffer.std_best_reward,
                            "eval/std episode length": self.buffer.std_episode_length,
                    }
                # convert everything to floating points
                for key, value in eval_dict.items():
                    if isinstance(value, torch.Tensor):
                        eval_dict[key]=value.item()
                self.run_results[-1].update(eval_dict)
                if self.use_wandb:
                    wandb.log(
                        data=eval_dict,
                        step=self.itr,
                        commit=False,
                    )
                
                if self.current_best_reward < self.buffer.avg_episode_reward:
                    self.current_best_reward = self.buffer.avg_episode_reward
                    self.is_best_so_far = True
                    log.info(f"New best reward evaluated: {self.current_best_reward:4.3f}")
            else:
                # Updated training log with avg ± std formatting
                train_prt_str_basic = (
                    f"itr {self.itr} | Total Step {self.cnt_train_step / 1e6:4.3f} M | Time: {time:8.3f}\n"
                    f"Env: {self.env_name} x {self.n_envs}\n"
                    f"Episode Reward: {self.buffer.avg_episode_reward:8.2f} ± {self.buffer.std_episode_reward:8.2f}\n"
                    f"Success Rate: {self.buffer.success_rate * 100:3.2f}% ± {self.buffer.std_success_rate * 100:3.2f}% \n"
                    f"Avg Best Reward: {self.buffer.avg_best_reward:8.2f} ± {self.buffer.std_best_reward:8.2f}\n"
                    f"Episode Length: {self.buffer.avg_episode_length:8.2f} ± {self.buffer.std_episode_length:8.2f}\n"
                    f"Actor lr :{self.actor_optimizer.param_groups[0]['lr']:.2e}\n"
                    f"Critic lr: {self.critic_optimizer.param_groups[0]['lr']:.2e}\n"
                )
                formatted_items = [f"{key}: {value:.3e}" for key, value in self.train_ret_dict.items()]
                num_items_per_row = 10
                for i in range(0, len(formatted_items), num_items_per_row):
                    train_prt_str_basic += " | ".join(formatted_items[i:i+num_items_per_row]) + "\n"
                log.info(train_prt_str_basic + train_prt_str_additional)
                
                # upload to wandb
                train_log_dict_basic = {
                    "train/total env step": self.cnt_train_step,
                    "train/success rate": self.buffer.success_rate,
                    "train/avg episode reward": self.buffer.avg_episode_reward,
                    "train/avg episode length": self.buffer.avg_episode_length,
                    "train/num episode": self.buffer.num_episode_finished,                    
                    "train/std success rate": self.buffer.std_success_rate,
                    "train/avg best reward": self.buffer.avg_best_reward,
                    "train/std episode reward": self.buffer.std_episode_reward,
                    "train/std best reward": self.buffer.std_best_reward,
                    "train/std episode length": self.buffer.std_episode_length,
                    "train/actor lr": self.actor_optimizer.param_groups[0]["lr"],
                    "train/critic lr": self.critic_optimizer.param_groups[0]["lr"],
                }
                loss_dict = {"loss/"+key: value for key, value in self.train_ret_dict.items()}
                train_log_dict_basic.update(loss_dict)
                train_log_dict = {**train_log_dict_basic, **(train_log_dict_additional or {})}
                # convert everything to floating points
                for key, value in train_log_dict.items():
                    if isinstance(value, torch.Tensor):
                        train_log_dict[key]=value.item()
                self.run_results[-1].update(train_log_dict)
                
                # Log training metrics to WandB
                if self.use_wandb:
                    wandb.log(
                        data=train_log_dict,
                        step=self.itr,
                        commit=True,
                    )
            # 保存每次迭代的数据到文本文件和 CSV
            self._save_metrics_to_file()
            with open(self.result_path, "wb") as f:
                pickle.dump(self.run_results, f)
                
    # def _save_metrics_to_file(self):
    #     # """保存训练指标到文本文件和 CSV 文件"""
    #     mode = "eval" if self.eval_mode else "train"
        
    #     # 获取损失信息（仅训练模式有）
    #     loss_info = ""
    #     csv_loss_values = ",,,,"  # loss, pg_loss, value_loss, entropy_loss, approx_kl, clip_frac
    #     if not self.eval_mode and hasattr(self, 'train_ret_dict'):
    #         entropy_loss_val = self.train_ret_dict.get('entropy_loss', 0)
    #         if hasattr(entropy_loss_val, 'item'):
    #             entropy_loss_val = entropy_loss_val.item()
    #         loss_info = (f"  Loss: {self.train_ret_dict.get('loss', 0):.4e}, "
    #         f"PG Loss: {self.train_ret_dict.get('pg loss', 0):.4e}, "
    #         f"Value Loss: {self.train_ret_dict.get('value loss', 0):.4e}, "
    #         f"Entropy Loss: {entropy_loss_val:.4e}")
    #         csv_loss_values = f"{self.train_ret_dict.get('loss', '')},{self.train_ret_dict.get('pg loss', '')},{self.train_ret_dict.get('value loss', '')},{self.train_ret_dict.get('entropy_loss', '')},{self.train_ret_dict.get('approx kl', '')},{self.train_ret_dict.get('clip_frac', '')}"
    #     # if not self.eval_mode:
    #     #     return
    #     # mode = "eval"
    #     # 写入文本文件（人类可读格式）
    #     with open(self.metrics_log_path, 'a') as f:
    #         f.write(f"[{mode.upper()}] Itr {self.itr} | Step {self.cnt_train_step} | "
    #                f"Success Rate: {self.buffer.success_rate*100:.2f}% | "
    #                f"Avg Reward: {self.buffer.avg_episode_reward:.2f} | "
    #                f"Avg Length: {self.buffer.avg_episode_length:.2f} | "
    #                f"Actor LR: {self.actor_optimizer.param_groups[0]['lr']:.2e} | "
    #                f"Critic LR: {self.critic_optimizer.param_groups[0]['lr']:.2e}"
    #                f"{loss_info}\n")
    #             # 写入文本文件（人类可读格式）
    #     # with open(self.metrics_log_path, 'a') as f:
    #     #     f.write(f"[{mode.upper()}] Itr {self.itr} | Step {self.cnt_train_step} | "
    #     #            f"Success Rate: {self.buffer.success_rate*100:.2f}% | "
    #     #            f"Avg Reward: {self.buffer.avg_episode_reward:.2f} | "
    #     #            f"Avg Length: {self.buffer.avg_episode_length:.2f} | "
    #     #            f"Actor LR: {self.actor_optimizer.param_groups[0]['lr']:.2e} | "
    #     #            f"Critic LR: {self.critic_optimizer.param_groups[0]['lr']:.2e}\n")
        
    #     # 写入 CSV 文件（便于后续分析和绘图）
    #     with open(self.metrics_csv_path, 'a') as f:
    #         f.write(f"{self.itr},{self.cnt_train_step},{mode},"
    #                f"{self.buffer.success_rate},{self.buffer.avg_episode_reward},"
    #                f"{self.buffer.avg_best_reward},{self.buffer.avg_episode_length},"
    #                f"{self.actor_optimizer.param_groups[0]['lr']},"
    #                f"{self.critic_optimizer.param_groups[0]['lr']},"
    #                f"{csv_loss_values}\n")
    #     # with open(self.metrics_csv_path, 'a') as f:
    #     #     f.write(f"{self.itr},{self.cnt_train_step},{mode},"
    #     #            f"{self.buffer.success_rate},{self.buffer.avg_episode_reward},"
    #     #            f"{self.buffer.avg_best_reward},{self.buffer.avg_episode_length},"
    #     #            f"{self.actor_optimizer.param_groups[0]['lr']},"
    #     #            f"{self.critic_optimizer.param_groups[0]['lr']}\n")    
    #         # 同时调用专门的评估CSV保存方法
    #     self.save_evaluation_to_csv()

    # def save_evaluation_to_csv(self):
    #     """保存评估结果到CSV文件，格式符合可视化脚本要求"""
    #     # 准备数据，格式为: Step, [方法名]_[指标]_seed_[种子]_evaluations
    #     step = self.cnt_train_step

    #     # 构建方法名 (使用 cfg 中的 name 或环境名)
    #     model_name = getattr(self, 'name', self.env_name)

    #     # 准备数据字典
    #     data = {
    #         'Step': [step],
    #         f'{model_name}_success_rate_seed_{self.seed}_evaluations': [self.buffer.success_rate],
    #         f'{model_name}_avg_episode_reward_seed_{self.seed}_evaluations': [self.buffer.avg_episode_reward],
    #         f'{model_name}_avg_best_reward_seed_{self.seed}_evaluations': [self.buffer.avg_best_reward],
    #         f'{model_name}_avg_episode_length_seed_{self.seed}_evaluations': [self.buffer.avg_episode_length],
    #         f'{model_name}_std_success_rate_seed_{self.seed}_evaluations': [self.buffer.std_success_rate],
    #         f'{model_name}_std_episode_reward_seed_{self.seed}_evaluations': [self.buffer.std_episode_reward],
    #         f'{model_name}_std_best_reward_seed_{self.seed}_evaluations': [self.buffer.std_best_reward],
    #         f'{model_name}_std_episode_length_seed_{self.seed}_evaluations': [self.buffer.std_episode_length],
    #     }

    #     # 确保目录存在
    #     eval_output_dir = os.path.join(self.logdir, 'eval_results')
    #     os.makedirs(eval_output_dir, exist_ok=True)

    #     # 保存或追加到CSV文件
    #     csv_path = os.path.join(eval_output_dir, f"{self.env_name}_evaluation.csv")
    #     df = pd.DataFrame(data)

    #     if os.path.exists(csv_path):
    #         existing_df = pd.read_csv(csv_path)
    #         # 检查是否已存在相同Step的数据，如果存在则更新，否则追加
    #         if step in existing_df['Step'].values:
    #             existing_df = existing_df[existing_df['Step'] != step]
    #         df = pd.concat([existing_df, df], ignore_index=True)
    #         df = df.sort_values('Step').reset_index(drop=True)

    #     df.to_csv(csv_path, index=False)
    #     log.info(f"Evaluation results saved to {csv_path}")
    
    def _save_metrics_to_file(self):
        """保存训练指标到文本文件和 CSV 文件"""
        mode = "eval" if self.eval_mode else "train"
        
        # 获取损失信息（仅训练模式有）
        loss_info = ""
        csv_loss_values = ",,,,"  # loss, pg_loss, value_loss, entropy_loss, approx_kl, clip_frac
        if not self.eval_mode and hasattr(self, 'train_ret_dict'):
            entropy_loss_val = self.train_ret_dict.get('entropy_loss', 0)
            if hasattr(entropy_loss_val, 'item'):
                entropy_loss_val = entropy_loss_val.item()
            loss_info = (f"  Loss: {self.train_ret_dict.get('loss', 0):.4e}, "
            f"PG Loss: {self.train_ret_dict.get('pg loss', 0):.4e}, "
            f"Value Loss: {self.train_ret_dict.get('value loss', 0):.4e}, "
            f"Entropy Loss: {entropy_loss_val:.4e}")
            csv_loss_values = f"{self.train_ret_dict.get('loss', '')},{self.train_ret_dict.get('pg loss', '')},{self.train_ret_dict.get('value loss', '')},{self.train_ret_dict.get('entropy_loss', '')},{self.train_ret_dict.get('approx kl', '')},{self.train_ret_dict.get('clip_frac', '')}"
        
        # 写入文本文件（人类可读格式）
        with open(self.metrics_log_path, 'a') as f:
            f.write(f"[{mode.upper()}] Itr {self.itr} | Step {self.cnt_train_step} | "
                   f"Success Rate: {self.buffer.success_rate*100:.2f}% | "
                   f"Avg Reward: {self.buffer.avg_episode_reward:.2f} | "
                   f"Avg Length: {self.buffer.avg_episode_length:.2f} | "
                   f"Actor LR: {self.actor_optimizer.param_groups[0]['lr']:.2e} | "
                   f"Critic LR: {self.critic_optimizer.param_groups[0]['lr']:.2e}"
                   f"{loss_info}\n")
        
        # 写入 CSV 文件（便于后续分析和绘图）
        with open(self.metrics_csv_path, 'a') as f:
            f.write(f"{self.itr},{self.cnt_train_step},{mode},"
                   f"{self.buffer.success_rate},{self.buffer.avg_episode_reward},"
                   f"{self.buffer.avg_best_reward},{self.buffer.avg_episode_length},"
                   f"{self.actor_optimizer.param_groups[0]['lr']},"
                   f"{self.critic_optimizer.param_groups[0]['lr']},"
                   f"{csv_loss_values}\n")