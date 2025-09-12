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



class InSampleAC(BaseAgentDoubleQSingleV):
    def __init__(self,
                 discrete_action: bool,
                 action_dim: int,
                 state_dim: int,
                 gamma: float,
                 batch_size: float,
                 alpha: float,
                 device: torch.device,
                 actor: torch.nn.Module,
                 behavior_policy: torch.nn.Module,
                 critic: torch.nn.Module,
                 replay_buffer: torch.nn.Module,
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
        self.behavior_policy = behavior_policy
        self.critic = critic
        self.buffer = replay_buffer


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
            sampled_next_actions, next_log_probs, _ = self.actor.sample(next_state_batch)
            next_q, _, _ = self.get_min_q_target(next_state_batch, sampled_next_actions)
            target_q_value = reward_batch + self.gamma * mask_batch * (next_q - self.alpha * next_log_probs)
            
            sampled_actions, log_probs, _ = self.actor.policy.sample(state_batch)
            min_q, _, _ = self.get_min_q_target(state_batch, sampled_actions)

        _, q_value_1, q_value_2 = self.get_min_q_value(state_batch, action_batch)
        q_loss_1 = F.mse_loss(target_q_value, q_value_1)
        q_loss_2 = F.mse_loss(target_q_value, q_value_2)
        q_loss = (q_loss_1 + q_loss_2) * 0.5
        self.critic.optimizer_q.zero_grad()
        q_loss.backward()
        self.critic.optimizer_q.step()

        v = self.get_v_value(state_batch)
        v_target = min_q - self.alpha * log_probs
        v_loss = F.mse_loss(v_target, v)

        self.critic.optimizer_v.zero_grad()
        v_loss.backward()
        self.critic.optimizer_v.step()
        #from IPython import embed; embed(); exit()
        return q_loss.detach().item(), v_loss.detach().item()
    

    def update_actor(self):
        state_batch, action_batch, _, _, _ = self.buffer.sample(batch_size=self.batch_size)
        # print(f"state {state_batch}, action {action_batch}")
        if state_batch is None:
            # Too few samples in the buffer to sample
            return
        with torch.no_grad():
            v = self.get_v_value(state_batch)
            min_q, _, _ = self.get_min_q_value(state_batch, action_batch)
            behavior_log_prob = self.behavior_policy.policy.log_prob(state_batch, action_batch)
        clipped = torch.clip(torch.exp((min_q - v) / self.alpha - behavior_log_prob), 1e-8, 10000)
        log_probs = self.actor.policy.log_prob(state_batch, action_batch)
        assert clipped.shape == log_probs.shape, f"weights shape {clipped.shape}, log_probs shape {log_probs.shape}"
        actor_loss = -(clipped * log_probs).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        behavior_logll = self.behavior_policy.policy.log_prob(state_batch, action_batch)
        behavior_loss = -behavior_logll.mean()
        # print("beh", beh_log_probs.size(), beh_loss)
        self.behavior_policy.optimizer.zero_grad()
        behavior_loss.backward()
        self.behavior_policy.optimizer.step()
        return actor_loss.detach().item()

    def reset(self) -> None:
        return

