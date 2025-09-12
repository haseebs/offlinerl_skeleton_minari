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



class SQL(BaseAgentDoubleQSingleV):
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
            next_v = self.get_v_value(next_state_batch)
            target_q_value = reward_batch + self.gamma * mask_batch * next_v
            min_q, _, _ = self.get_min_q_target(state_batch, action_batch)

        _, q_value_1, q_value_2 = self.get_min_q_value(state_batch, action_batch)
        q_loss_1 = F.mse_loss(target_q_value, q_value_1)
        q_loss_2 = F.mse_loss(target_q_value, q_value_2)
        q_loss = (q_loss_1 + q_loss_2) * 0.5
        # Calculate the loss on the critic
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        # Update the critic
        self.critic.optimizer_q.zero_grad()
        q_loss.backward()
        self.critic.optimizer_q.step()

        v = self.get_v_value(state_batch)
        sp_term = (min_q - v) / (2 * self.alpha) + 1.0
        sp_weight = torch.where(sp_term > 0, 1., 0.)
        v_loss = (sp_weight * (sp_term ** 2) + v / self.alpha).mean()

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
            q, _, _ = self.get_min_q_target(state_batch, action_batch)
        weights = q - v
        weights = torch.clip(weights, 0., 100.0)
        log_probs = self.actor.policy.log_prob(state_batch, action_batch)
        assert weights.shape == log_probs.shape, f"weights shape {weights.shape}, log_probs shape {log_probs.shape}"
        actor_loss = -(weights * log_probs).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
        return actor_loss.detach().item()

    def reset(self) -> None:
        return

