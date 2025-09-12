from abc import ABC, abstractmethod
from jaxtyping import Float
import numpy as np
import torch
from .base_agent import BaseAgentDoubleQSingleV
import torch.nn.functional as F
import os
import sys
sys.path.append("../utils")
sys.path.append("../models")
from utils.nn_utils import hard_update
from torch.nn.utils import clip_grad_norm_



class IQL(BaseAgentDoubleQSingleV):
    def __init__(self,
                 discrete_action: bool,
                 action_dim: int,
                 state_dim: int,
                 gamma: float,
                 batch_size: float,
                 alpha: float,
                 device: torch.device,
                 actor: torch.nn.Module,
                 critic: torch.nn.Module,
                 replay_buffer: torch.nn.Module,
                 expectile: float,
                 ) -> None:
        super().__init__()
        self.discrete_action = discrete_action
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.alpha = alpha
        self.device = device
        self.actor = actor
        self.critic = critic
        self.buffer = replay_buffer

        self.temperature = alpha
        self.expectile = expectile #torch.FloatTensor([0.8]).to(device)
        self.clip_grad_param = 100

    @torch.no_grad()
    def act(self, state: Float[np.ndarray, "state_dim"], greedy: bool=False) -> Float[np.ndarray, "action_dim"]:
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _, action_mean = self.actor.sample(state)
        act = action.detach().cpu().numpy()[0]
        if greedy:
            act = action_mean.detach().cpu().numpy()[0]
        if not self.discrete_action:
            return act
        else:
            return int(act[0])

    def expectile_loss(self, diff, expectile=0.8):
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        # print(f"weight {weight.shape}, diff {diff.shape}")
        return weight * (diff ** 2)


    def update_critic(self):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, \
            mask_batch = self.buffer.sample(batch_size=self.batch_size)
        # print(f"mask batch {mask_batch}")
        if state_batch is None:
            # Too few samples in the buffer to sample
            return
        # When updating Q functions, we don't want to backprop through the
        # policy and target network parameters
        with torch.no_grad():
            next_v = self.get_v_value(next_state_batch)
            target_q_value = reward_batch + self.gamma * mask_batch * next_v
            min_q_value_expectile, _, _ = self.get_min_q_target(state_batch, action_batch)

        _, q_value_1, q_value_2 = self.get_min_q_value(state_batch, action_batch)
        q_loss_1 = F.mse_loss(target_q_value, q_value_1)
        q_loss_2 = F.mse_loss(target_q_value, q_value_2)
        q_loss = (q_loss_1 + q_loss_2) * 0.5
        # Calculate the loss on the critic
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        # Update the critic
        self.critic.optimizer_q.zero_grad()
        q_loss.backward()
        clip_grad_norm_(self.critic.q_value_net.parameters(), self.clip_grad_param)
        self.critic.optimizer_q.step()

        v_value = self.get_v_value(state_batch)
        v_loss = self.expectile_loss(min_q_value_expectile - v_value, self.expectile).mean()
        self.critic.optimizer_v.zero_grad()
        v_loss.backward()
        self.critic.optimizer_v.step()
        #from IPython import embed; embed(); exit()
        return q_loss.detach().item(), v_loss.detach().item()

    def update_actor(self):
        state_batch, action_batch, reward_batch, next_state_batch, \
            mask_batch = self.buffer.sample(batch_size=self.batch_size)
        # print(f"state {state_batch}, action {action_batch}")
        if state_batch is None:
            # Too few samples in the buffer to sample
            return
        with torch.no_grad():
            v = self.get_v_value(state_batch)
            q, _, _ = self.get_min_q_target(state_batch, action_batch)
        exp_adv = torch.exp((q - v) / self.temperature)
        exp_adv = torch.min(exp_adv, torch.FloatTensor([100.0]).to(state_batch.device))
        log_probs = self.actor.policy.log_prob(state_batch, action_batch)
        assert exp_adv.shape == log_probs.shape, f"exp_adv shape {exp_adv.shape}, log_probs shape {log_probs.shape}"
        actor_loss = -(exp_adv * log_probs).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
        return actor_loss.detach().item()

    def reset(self) -> None:
        return


    # def compute_loss_pi(self, data):
    #     states, actions = data['obs'], data['act']
    #     with torch.no_grad():
    #         v = self.value_net(states)
    #     min_Q, _, _ = self.get_q_value_target(states, actions)
    #     exp_a = torch.exp((min_Q - v) * self.temperature)
    #     exp_a = torch.min(exp_a, torch.FloatTensor([100.0]).to(states.device))
    #     log_probs = self.ac.pi.log_prob(states, actions)
    #     actor_loss = -(exp_a * log_probs).mean()
    #     # print("pi", min_Q.size(), v.size(), exp_a.size(), log_probs.size())
    #     return actor_loss, log_probs

    # def compute_loss_value(self, data):
    #     states, actions = data['obs'], data['act']
    #     min_Q, _, _ = self.get_q_value_target(states, actions)

    #     value = self.value_net(states)
    #     value_loss = self.expectile_loss(min_Q - value, self.expectile).mean()
    #     # print("value", min_Q.size(), value.size(), self.expectile_loss(min_Q - value, self.expectile).size())
    #     return value_loss



    # def compute_loss_q(self, data):
    #     states, actions, rewards, next_states, dones = data['obs'], data['act'], data['reward'], data['obs2'], data['done']
    #     with torch.no_grad():
    #         next_v = self.value_net(next_states)
    #         q_target = rewards + (self.gamma * (1 - dones) * next_v)

    #     _, q1, q2 = self.get_q_value(states, actions, with_grad=True)
    #     critic1_loss = (0.5* (q_target - q1) ** 2).mean()
    #     critic2_loss = (0.5* (q_target - q2) ** 2).mean()
    #     loss_q = (critic1_loss + critic2_loss) * 0.5
    #     # print("q", q1.shape, q2.shape, q_target.shape, rewards.shape, dones.shape, actions.shape)
    #     return loss_q

    # def update(self, data):
    #     self.value_optimizer.zero_grad()
    #     loss_vs = self.compute_loss_value(data)
    #     loss_vs.backward()
    #     self.value_optimizer.step()

    #     loss_q = self.compute_loss_q(data)
    #     self.q_optimizer.zero_grad()
    #     loss_q.backward()
    #     self.q_optimizer.step()

    #     if self.use_target_network and self.total_steps % self.target_network_update_freq == 0:
    #         self.sync_target()

    #     loss_pi, _ = self.compute_loss_pi(data)
    #     self.pi_optimizer.zero_grad()
    #     loss_pi.backward()
    #     self.pi_optimizer.step()

    #     return


    # def save(self, timestamp=''):
    #     parameters_dir = self.parameters_dir
    #     params = {
    #         "actor_net": self.ac.pi.state_dict(),
    #         "critic_net": self.ac.q1q2.state_dict(),
    #         "value_net": self.value_net.state_dict()
    #     }
    #     path = os.path.join(parameters_dir, "parameter"+timestamp)
    #     torch.save(params, path)

    # def load(self, parameters_dir, timestamp=''):
    #     path = os.path.join(parameters_dir, "parameter"+timestamp)
    #     model = torch.load(path)
    #     self.ac.pi.load_state_dict(model["actor_net"])
    #     self.ac.q1q2.load_state_dict(model["critic_net"])
    #     self.value_net.load_state_dict(model["value_net"])


# from collections import namedtuple
# class IQLqG(IQL):
#     def __init__(self, cfg):
#         super(IQLqG, self).__init__(cfg)
#         self.beh_pi = Student(cfg.device, cfg.state_dim, cfg.action_dim, [cfg.hidden_units] * 2,
#                               cfg.action_min, cfg.action_max, df=cfg.distribution_param)
#         self.beh_pi_optimizer = torch.optim.Adam(list(self.beh_pi.parameters()), cfg.pi_lr)

#         self.test_mode = cfg.test_mode
#         if self.test_mode: # student T
#             pi = Student(cfg.device, cfg.state_dim, cfg.action_dim, [cfg.hidden_units] * 2,
#                                   cfg.action_min, cfg.action_max, df=cfg.distribution_param)
#             q1q2 = self.get_critic_func(cfg.discrete_control, cfg.device, cfg.state_dim, cfg.action_dim, cfg.hidden_units)
#             AC = namedtuple('AC', ['q1q2', 'pi'])
#             self.ac = AC(q1q2=q1q2, pi=pi)
#             pi_target = Student(cfg.device, cfg.state_dim, cfg.action_dim, [cfg.hidden_units] * 2,
#                                   cfg.action_min, cfg.action_max, df=cfg.distribution_param)
#             q1q2_target = self.get_critic_func(cfg.discrete_control, cfg.device, cfg.state_dim, cfg.action_dim, cfg.hidden_units)
#             q1q2_target.load_state_dict(q1q2.state_dict())
#             pi_target.load_state_dict(pi.state_dict())
#             ACTarg = namedtuple('ACTarg', ['q1q2', 'pi'])
#             self.ac_targ = ACTarg(q1q2=q1q2_target, pi=pi_target)
#             self.ac_targ.q1q2.load_state_dict(self.ac.q1q2.state_dict())
#             self.ac_targ.pi.load_state_dict(self.ac.pi.state_dict())
#             self.pi_optimizer = torch.optim.Adam(list(self.ac.pi.parameters()), cfg.pi_lr)
#             self.q_optimizer = torch.optim.Adam(list(self.ac.q1q2.parameters()), cfg.q_lr)

#     def compute_loss_beh_pi(self, data):
#         states, actions = data['obs'], data['act']
#         beh_log_probs = self.beh_pi.log_prob(states, actions)
#         beh_loss = -beh_log_probs.mean()
#         return beh_loss, beh_log_probs

#     def compute_loss_pi(self, data):
#         states = data['obs']
#         actions, log_pi = self.ac.pi.rsample(states)
#         with torch.no_grad():
#             log_pi_beta = self.beh_pi.log_prob(states, actions)
#             v = self.value_net(states)
#         min_Q, _, _ = self.get_q_value_target(states, actions)
#         adv = (min_Q - v) * self.temperature
#         if self.test_mode:
#             pi_loss = (log_pi - log_pi_beta - adv).mean()
#         else:
#             nonzeros = torch.where(log_pi_beta > -6.)[0]
#             pi_loss = (log_pi - log_pi_beta - adv)[nonzeros].sum()/len(log_pi)
#         # print(log_pi.mean(), log_pi_beta.mean(), adv.mean(), pi_loss)
#         # print("pi", log_pi.size(), log_pi_beta.size(), min_Q.size(), v.size())
#         return pi_loss, log_pi

#     def update(self, data):
#         loss_beh_pi, _ = self.compute_loss_beh_pi(data)
#         self.beh_pi_optimizer.zero_grad()
#         loss_beh_pi.backward()
#         self.beh_pi_optimizer.step()

#         self.value_optimizer.zero_grad()
#         loss_vs = self.compute_loss_value(data)
#         loss_vs.backward()
#         self.value_optimizer.step()

#         loss_q = self.compute_loss_q(data)
#         self.q_optimizer.zero_grad()
#         loss_q.backward()
#         self.q_optimizer.step()

#         loss_pi, _ = self.compute_loss_pi(data)
#         self.pi_optimizer.zero_grad()
#         loss_pi.backward()
#         self.pi_optimizer.step()

#         if self.use_target_network and self.total_steps % self.target_network_update_freq == 0:
#             self.sync_target()
#         return

#     def save(self, timestamp=''):
#         parameters_dir = self.parameters_dir
#         params = {
#             "actor_net": self.ac.pi.state_dict(),
#             "critic_net": self.ac.q1q2.state_dict(),
#             "value_net": self.value_net.state_dict(),
#             "behavior_net": self.beh_pi.state_dict()
#         }
#         path = os.path.join(parameters_dir, "parameter"+timestamp)
#         torch.save(params, path)
