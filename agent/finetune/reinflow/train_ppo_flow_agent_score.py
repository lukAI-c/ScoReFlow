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
fine-tuning.
"""
import os
import logging
log = logging.getLogger(__name__)
from tqdm import tqdm as tqdm
import numpy as np
import pandas as pd
import torch
from agent.finetune.reinflow.train_ppo_agent import TrainPPOAgent
from model.flow.ft_ppo.ppoflow_score import PPOFlow
from agent.finetune.reinflow.buffer import PPOFlowBuffer
import matplotlib.pyplot as plt
# define buffer on cpu or cuda. Currently GPU version is not offering significant acceleration...
# communication could be a bottleneck, it now just increases GPU volatile utilization from 7% to 13%
# this could own to mujoco generating data on cpu and we frequently moves them to and from GPUs. 


# this script works for both pretrained 1-ReFlow and ShortCutFlows.



class TrainPPOFlowAgent(TrainPPOAgent):
    """
    Score-based PPO Flow Agent

    使用分数函数进行探索，无需训练额外的噪声网络:
    其中:
        vt = bt(x) 速度场
        st = (t * bt(x) - x) / (1 - t) 分数函数
        ε ~ N(0, I) 标准高斯噪声
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        # Reward horizon --- always set to act_steps for now
        self.skip_initial_eval = cfg.get('skip_initial_eval', False)
        self.reward_horizon = cfg.get("reward_horizon", self.act_steps)
        self.inference_steps = self.model.inference_steps
        self.repeat_samples = cfg.train.get("repeat_samples", False)

        self.normalize_act_space_dim = True
        self.normalize_denoising_horizon = True
        self.lr_schedule = cfg.train.lr_schedule
        self.clip_intermediate_actions = cfg.train.get("clip_intermediate_actions", True)
        self.account_for_initial_stochasticity = cfg.train.get('account_for_initial_stochasticity', True)
        if self.lr_schedule not in ["fixed", "adaptive_kl"]:
            raise ValueError("lr_schedule should be 'fixed' or 'adaptive_kl'")
        self.actor_lr = cfg.train.actor_lr
        self.critic_lr = cfg.train.critic_lr

        self.model: PPOFlow

        # Score-based exploration parameters
        self.epsilon_t = self.model.epsilon_t
        self.epsilon_schedule = self.model.epsilon_schedule

        # Log epsilon schedule
        log.info(f"Score-based exploration: epsilon_t={self.epsilon_t}, schedule={self.epsilon_schedule}")

        # Plot epsilon schedule
        # epsilon_values = self._compute_epsilon_schedule()
        # plt.figure()
        # plt.plot(np.arange(self.inference_steps), epsilon_values)
        # plt.xlabel('Inference Step')
        # plt.ylabel('Epsilon')
        # plt.title(f'Epsilon Schedule: {self.epsilon_schedule}')
        name = os.path.join(self.logdir, 'epsilon_schedule') + '.png'
        # plt.savefig(name)
        # plt.close()
        log.info(f"Epsilon schedule saved to {name}")

        self.initial_ratio_error_threshold = 1e-6

    # def _compute_epsilon_schedule(self):
    #     """计算 epsilon schedule 用于可视化"""
    #     epsilon_values = []
    #     for i in range(self.inference_steps):
    #         t = i / self.inference_steps
    #         if self.epsilon_schedule == 'constant':
    #             eps = self.epsilon_t
    #         elif self.epsilon_schedule == 'linear_decay':
    #             eps = self.epsilon_t * (1 - t)
    #         elif self.epsilon_schedule == 'cosine':
    #             eps = self.epsilon_t * 0.5 * (1 + np.cos(np.pi * t))
    #         else:
    #             eps = self.epsilon_t
    #         epsilon_values.append(eps)
    #     return epsilon_values

    def init_buffer(self):
        self.buffer = PPOFlowBuffer(
            n_steps=self.n_steps,
            n_envs=self.n_envs,
            n_ft_denoising_steps= self.inference_steps, 
            horizon_steps=self.horizon_steps,
            act_steps=self.act_steps,
            action_dim=self.action_dim,
            n_cond_step=self.n_cond_step,
            obs_dim=self.obs_dim,
            save_full_observation=self.save_full_observations,
            furniture_sparse_reward=self.furniture_sparse_reward,
            best_reward_threshold_for_success=self.best_reward_threshold_for_success,
            reward_scale_running=self.reward_scale_running,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            reward_scale_const=self.reward_scale_const,
            device=self.device,
        ) 
    
    def resume_training(self):
        """Resume training from checkpoint (score-based, no noise network to update)"""
        super().resume_training()
        log.info(f"Resumed training with score-based exploration: epsilon_t={self.epsilon_t}, schedule={self.epsilon_schedule}")
    
    def run(self):
        self.init_buffer()
        self.prepare_run()
        self.buffer.reset() # as long as we put items at the right position in the buffer (determined by 'step'), the buffer automatically resets when new iteration begins (step =0). so we only need to reset in the beginning. This works only for PPO buffer, otherwise may need to reset when new iter begins.
        if self.resume:
            self.resume_training()
        while self.itr < self.n_train_itr:
            self.prepare_video_path()
            self.set_model_mode()
            self.reset_env() # for gpu version, add device=self.device
            self.buffer.update_full_obs()
            for step in range(self.n_steps):
                
                with torch.no_grad():
                    cond = {
                        "state": torch.tensor(self.prev_obs_venv["state"], device=self.device, dtype=torch.float32)
                    }
                    value_venv = self.get_value(cond=cond) # for gpu version add , device=self.device
                    action_samples, chains_venv, logprob_venv = self.get_samples_logprobs(cond=cond, 
                                                                                          normalize_denoising_horizon=self.normalize_denoising_horizon,
                                                                                          normalize_act_space_dimension=self.normalize_act_space_dim, 
                                                                                          clip_intermediate_actions=self.clip_intermediate_actions,
                                                                                          account_for_initial_stochasticity=self.account_for_initial_stochasticity) # for gpu version, add , device=self.device
                
                # Apply multi-step action
                action_venv = action_samples[:, : self.act_steps]
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
                
                self.buffer.save_full_obs(info_venv)
                self.buffer.add(step, self.prev_obs_venv["state"], chains_venv, reward_venv, terminated_venv, truncated_venv, value_venv, logprob_venv)
                
                self.prev_obs_venv = obs_venv
                self.cnt_train_step+= self.n_envs * self.act_steps if not self.eval_mode else 0
            self.buffer.summarize_episode_reward()
            
            if not self.eval_mode:
                self.buffer.update(obs_venv, self.model.critic) # for gpu version, add device=self.device
                self.agent_update(verbose=self.verbose)
            
            # self.plot_state_trajecories() #(only in D3IL)
            
            self.log()                                          # diffusion_min_sampling_std
            self.update_lr()
            self.update_ent_coef_schedule()  # 更新熵系数调度
            self.adjust_finetune_schedule()# update finetune scheduler of ReFlow Policy
            self.save_model()
            self.itr += 1 
            
    def adjust_finetune_schedule(self):
        """Score-based: no noise network to adjust, epsilon schedule is fixed"""
        pass
        
    # overload...
    def save_model(self, only_save_policy_network=False):
        """
        saves model to disk; no ema recorded because we are doing RLFT.
        Score-based version: actor_ft is directly FlowMLP (no .policy wrapper)
        """
        # actor_ft 现在直接是 FlowMLP，不再有 .policy 包装
        policy_network_state_dict = {
            'network.' + key: value for key, value in self.model.actor_ft.state_dict().items()
        }
        
        if only_save_policy_network:
            data = {
                "itr": self.itr,
                "cnt_train_steps": self.cnt_train_step,
                "policy": policy_network_state_dict,
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "actor_lr_scheduler": self.actor_lr_scheduler.state_dict(),
                "critic_lr_scheduler": self.critic_lr_scheduler.state_dict(),
            }
        else:
            data = {
                "itr": self.itr,
                "cnt_train_steps": self.cnt_train_step,
                "model": self.model.state_dict(),  # for resume training
                "policy": policy_network_state_dict,  # flow policy for evaluation, without critic and exploration noise nets
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "actor_lr_scheduler": self.actor_lr_scheduler.state_dict(),
                "critic_lr_scheduler": self.critic_lr_scheduler.state_dict(),
            }
        
        # always save the last model for resume of training. 
        save_path = os.path.join(self.checkpoint_dir,f"last.pt")
        torch.save(data, os.path.join(self.checkpoint_dir, save_path))
        
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
    
    @torch.no_grad()
    def get_samples_logprobs(self, 
                             cond:dict, 
                             ret_device='cpu', 
                             save_chains=True, 
                             normalize_denoising_horizon=False, 
                             normalize_act_space_dimension=False, 
                             clip_intermediate_actions=True,
                             account_for_initial_stochasticity=True):
        # returns: action_samples are still numpy because mujoco engine receives np.
        if save_chains:
            action_samples, chains_venv, logprob_venv  = self.model.get_actions(cond, 
                                                                                eval_mode=self.eval_mode, 
                                                                                save_chains=save_chains, 
                                                                                normalize_denoising_horizon=normalize_denoising_horizon, 
                                                                                normalize_act_space_dimension=normalize_act_space_dimension, 
                                                                                clip_intermediate_actions=clip_intermediate_actions,
                                                                                account_for_initial_stochasticity=account_for_initial_stochasticity)        # n_envs , horizon_steps , act_dim
            return action_samples.cpu().numpy(), chains_venv.cpu().numpy() if ret_device=='cpu' else chains_venv, logprob_venv.cpu().numpy()  if ret_device=='cpu' else logprob_venv
        else:
            action_samples, logprob_venv  = self.model.get_actions(cond, 
                                                                   eval_mode=self.eval_mode, 
                                                                   save_chains=save_chains, 
                                                                   normalize_denoising_horizon=normalize_denoising_horizon, 
                                                                   normalize_act_space_dimension=normalize_act_space_dimension, 
                                                                   clip_intermediate_actions=clip_intermediate_actions,
                                                                   account_for_initial_stochasticity=account_for_initial_stochasticity)
            return action_samples.cpu().numpy(), logprob_venv.cpu().numpy()  if ret_device=='cpu' else logprob_venv
    
    def get_value(self, cond:dict, device='cpu'):
        # cond contains a floating-point torch.tensor on self.device
        if device == 'cpu':
            value_venv = self.model.critic.forward(cond).cpu().numpy().flatten()
        else:
            value_venv = self.model.critic.forward(cond).squeeze().float().to(self.device)
        return value_venv
    
    # overload
    def update_lr(self, val_metric=None):
        if self.target_kl and self.lr_schedule == 'adaptive_kl':   # adapt learning rate according to kl divergence on each minibatch.
            return
        else: # use predefined lr scheduler. 
            super().update_lr()
    
    def update_lr_adaptive_kl(self, approx_kl):
        min_actor_lr = 1e-5
        max_actor_lr = 5e-4
        min_critic_lr = 1e-5
        max_critic_lr = 1e-3
        tune='maintains'
        if approx_kl > self.target_kl * 2.0:
            self.actor_lr = max(min_actor_lr, self.actor_lr / 1.5)
            self.critic_lr = max(min_critic_lr, self.critic_lr / 1.5)
            tune = 'decreases'
        elif 0.0 < approx_kl and approx_kl < self.target_kl / 2.0:
            self.actor_lr = min(max_actor_lr, self.actor_lr * 1.5)
            self.critic_lr = min(max_critic_lr, self.critic_lr * 1.5)
            tune = 'increases'
        for actor_param_group, critic_param_group in zip(self.actor_optimizer.param_groups, self.critic_optimizer.param_groups):
            actor_param_group["lr"] = self.actor_lr
            critic_param_group["lr"] = self.critic_lr
        log.info(f"""adaptive kl {tune} lr: actor_lr={self.actor_optimizer.param_groups[0]["lr"]:.2e}, critic_lr={self.critic_optimizer.param_groups[0]["lr"]:.2e}""")
    
    def minibatch_generator(self):
        self.approx_kl = 0.0
        
        obs, chains, returns, oldvalues, advantages, oldlogprobs =  self.buffer.make_dataset()
        # Explained variation of future rewards using value function
        self.explained_var = self.buffer.get_explained_var(oldvalues, returns)
        
        self.total_steps = self.n_steps * self.n_envs
        for update_epoch in range(self.update_epochs):
            self.kl_change_too_much = False
            indices = torch.randperm(self.total_steps, device=self.device)
            if self.lr_schedule=='fixed' and self.kl_change_too_much:
                break
            for batch_id, start in enumerate(range(0, self.total_steps, self.batch_size)):
                end = start + self.batch_size
                inds_b = indices[start:end]
                minibatch = (
                    {"state": obs[inds_b]},
                    chains[inds_b],
                    returns[inds_b], 
                    oldvalues[inds_b],
                    advantages[inds_b],
                    oldlogprobs[inds_b] 
                )
                if self.lr_schedule=='fixed' and self.target_kl and self.approx_kl > self.target_kl: # we can also use adaptive KL instead of early stopping.
                    self.kl_change_too_much = True
                    log.warning(f"KL change too much, approx_kl ={self.approx_kl} > {self.target_kl} = target_kl, stop optimization.")
                    break
                
                yield update_epoch, batch_id, minibatch    

    def minibatch_generator_repeat(self):
        self.approx_kl = 0.0
        
        obs, chains, returns, oldvalues, advantages, oldlogprobs =  self.buffer.make_dataset()
        # Explained variation of future rewards using value function
        self.explained_var = self.buffer.get_explained_var(oldvalues, returns)
        
        duplicate_multiplier = 10   #self.ft_denoising_steps of PPO diffusion. this is added to strictly align with the batchsize of PPODiffusion.
        
        self.total_steps = self.n_steps * self.n_envs *  duplicate_multiplier
        
        for update_epoch in range(self.update_epochs):
            self.kl_change_too_much = False
            indices = torch.randperm(self.total_steps, device=self.device)
            if self.lr_schedule=='fixed' and self.kl_change_too_much:
                break
            for batch_id, start in enumerate(range(0, self.total_steps, self.batch_size)):
                end = start + self.batch_size
                inds_b = indices[start:end]
                batch_inds_b, denoising_inds_b = torch.unravel_index(
                    inds_b,
                    (self.n_steps * self.n_envs, duplicate_multiplier),
                )
                minibatch = (
                    {"state": obs[batch_inds_b]},
                    chains[batch_inds_b],
                    returns[batch_inds_b], 
                    oldvalues[batch_inds_b],
                    advantages[batch_inds_b],
                    oldlogprobs[batch_inds_b] 
                )
                if self.lr_schedule=='fixed' and self.target_kl and self.approx_kl > self.target_kl: # we can also use adaptive KL instead of early stopping.
                    self.kl_change_too_much = True
                    log.warning(f"KL change too much, approx_kl ={self.approx_kl} > {self.target_kl} = target_kl, stop optimization.")
                    break
                
                yield update_epoch, batch_id, minibatch

    def agent_update(self, verbose=True):
        clipfracs_list = []
        noise_std_list = []
        for update_epoch, batch_id, minibatch in self.minibatch_generator() if not self.repeat_samples else self.minibatch_generator_repeat():

            # minibatch gradient descent
            self.model: PPOFlow
            
            # print(f"minibatch contains {minibatch[0]['state'].shape}. self.n_envs={self.n_envs}")
            pg_loss, entropy_loss, v_loss, bc_loss, \
            clipfrac, approx_kl, ratio, \
            oldlogprob_min, oldlogprob_max, oldlogprob_std, \
                newlogprob_min, newlogprob_max, newlogprob_std, \
                noise_std, Q_values= self.model.loss(*minibatch, 
                                                    use_bc_loss=self.use_bc_loss, 
                                                    bc_loss_type=self.bc_loss_type, normalize_denoising_horizon=self.normalize_denoising_horizon, 
                                                    normalize_act_space_dimension=self.normalize_act_space_dim,
                                                    verbose=verbose,
                                                    clip_intermediate_actions=self.clip_intermediate_actions,
                                                    account_for_initial_stochasticity=self.account_for_initial_stochasticity)
            self.approx_kl = approx_kl
            if verbose:
                log.info(f"update_epoch={update_epoch}/{self.update_epochs}, batch_id={batch_id}/{max(1, self.total_steps // self.batch_size)}, ratio={ratio:.3f}, clipfrac={clipfrac:.3f}, approx_kl={self.approx_kl:.2e}")
            
            if update_epoch ==0  and batch_id ==0 and np.abs(ratio-1.00)> self.initial_ratio_error_threshold:
                raise ValueError(f"ratio={ratio} not 1.00 when update_epoch ==0  and batch_id ==0, there must be some bugs in your code not related to hyperparameters !")
            
            if self.target_kl and self.lr_schedule == 'adaptive_kl':
                self.update_lr_adaptive_kl(self.approx_kl)
            
            loss = pg_loss + entropy_loss * self.ent_coef + v_loss * self.vf_coef + bc_loss * self.bc_coeff
            
            clipfracs_list += [clipfrac]
            noise_std_list += [noise_std]
            
            # update policy and critic
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            loss.backward()
            
            # debug the losses
            actor_norm = torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), max_norm=float('inf'))
            actor_old_norm = torch.nn.utils.clip_grad_norm_(self.model.actor_old.parameters(), max_norm=float('inf'))
            critic_norm = torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), max_norm=float('inf'))
            if verbose:
                log.info(f"before clipping: actor_norm={actor_norm:.2e}, critic_norm={critic_norm:.2e}, actor_old_norm={actor_old_norm:.2e}")
            
            # always and frequently update critic
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            
            # after critic warmup to make the value estimate a reasonable value, update the actor less frequently but more times. 
            if self.itr >= self.n_critic_warmup_itr:
                if (self.itr-self.n_critic_warmup_itr) % self.actor_update_freq ==0:
                    for _ in range(self.actor_update_epoch):
                        if self.max_grad_norm:
                            torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), self.max_grad_norm)
                        self.actor_optimizer.step()
        
        clip_fracs = np.mean(clipfracs_list)
        score_stds = np.mean(noise_std_list)  # renamed: now represents score std, not noise std
        self.train_ret_dict = {
                "loss": loss,
                "pg loss": pg_loss,
                "value loss": v_loss,
                "entropy_loss": entropy_loss,
                "bc_loss": bc_loss,
                "approx kl": self.approx_kl,
                "ratio": ratio,
                "clipfrac": clip_fracs,
                "explained variance": self.explained_var,
                "old_logprob_min": oldlogprob_min,
                "old_logprob_max": oldlogprob_max,
                "old_logprob_std": oldlogprob_std,
                "new_logprob_min": newlogprob_min,
                "new_logprob_max": newlogprob_max,
                "new_logprob_std": newlogprob_std,
                "actor_norm": actor_norm,
                "critic_norm": critic_norm,
                "actor lr": self.actor_optimizer.param_groups[0]["lr"],
                "critic lr": self.critic_optimizer.param_groups[0]["lr"],
                "epsilon_t": self.model.epsilon_t,           # score-based: epsilon coefficient
                # "epsilon_schedule": self.model.epsilon_schedule,
                "score_std": score_stds,
                "gamma": self.model.gamma_score,# score std (|st|)
                "Q_values": Q_values
            }
    
    