from abc import ABC, abstractmethod
from jaxtyping import Float
import numpy as np
import torch
from .base_agent import BaseAgentDoubleQSingleV
import torch.nn.functional as F
import os
import sys
sys.path.append("../utils")
from utils.nn_utils import hard_update
from utils.utils import logq_x, expq_x


class TAWAC(BaseAgentDoubleQSingleV):
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
                 entropic_index: float,
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

        self.alpha = alpha # inversed
        self.entropic_index = entropic_index #torch.FloatTensor([0.8]).to(device)
        self.eps = 1e-8
        self.exp_threshold = 10000

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
        if state_batch is None:
            # Too few samples in the buffer to sample
            return
        # When updating Q functions, we don't want to backprop through the
        # policy and target network parameters
        next_state_action, _, _ = self.actor.policy.sample(next_state_batch)
        with torch.no_grad():
            next_q, _, _ = self.get_min_q_target(next_state_batch, next_state_action)
            target_q_value = reward_batch + mask_batch * self.gamma * next_q
            current_min_q, _, _ = self.get_min_q_target(state_batch, action_batch)
        """we use QV critic or double critic that has outputs two Q values"""
        _, q_value_1, q_value_2 = self.get_min_q_value(state_batch, action_batch)
        # Calculate the loss on the critic
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        # q_loss = F.mse_loss(target_q_value, q_value)
        q_loss_1 = F.mse_loss(target_q_value, q_value_1)
        q_loss_2 = F.mse_loss(target_q_value, q_value_2)
        q_loss = (q_loss_1 + q_loss_2) * 0.5
        # Update the critic
        self.critic.optimizer_q.zero_grad()
        q_loss.backward()
        self.critic.optimizer_q.step()
        # print(f"q loss {q_loss.detach().item()}")

        v_value = self.get_v_value(state_batch)
        value_loss = F.mse_loss(current_min_q, v_value)
        self.critic.optimizer_v.zero_grad()
        value_loss.backward()
        self.critic.optimizer_v.step()
        #from IPython import embed; embed(); exit()
        return q_loss.detach().item(), value_loss.detach().item()

    def update_actor(self):
        state_batch, action_batch, reward_batch, next_state_batch, \
            mask_batch = self.buffer.sample(batch_size=self.batch_size)
        if state_batch is None:
            # Too few samples in the buffer to sample
            return
        with torch.no_grad():
            min_q, _, _ = self.get_min_q_target(state_batch, action_batch)
            v = self.get_v_value(state_batch)

        assert self.entropic_index < 1.0, "q-exponential index should be less than 1 for filtering"
        expq_adv = expq_x((min_q - v) / self.alpha, self.entropic_index)
        # expq_adv = torch.min(expq_adv, torch.FloatTensor([100.0]).to(state_batch.device))
        expq_adv = torch.clip(expq_adv, min=self.eps, max=self.exp_threshold)

        log_probs = self.actor.policy.log_prob(state_batch, action_batch)
        assert expq_adv.shape == log_probs.shape, f"expq_adv shape {expq_adv.shape} log_probs shape {log_probs.shape}"
        actor_loss = -(expq_adv * log_probs).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
        return actor_loss.detach().item()

    def reset(self) -> None:
        return
