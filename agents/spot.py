from abc import ABC, abstractmethod
import logging
from jaxtyping import Float
import numpy as np
import torch
import torch.nn.functional as F
import os
import sys
from .base_agent import BaseAgentDoubleQ
sys.path.append("../utils")
sys.path.append("../models")
from models.vae import VAE
from torch.optim import Adam

log = logging.getLogger(__name__)

class SPOT(BaseAgentDoubleQ):
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
                beta: float,
                max_action: float,
                vae_latent_dim: int, 
                vae_optimizer: str,
                vae_lr: float,
                vae_num_samples: int,
                train_vae: bool = True,
                vae_train_iter: int = 1e5,
                  ):
        super(SPOT, self).__init__()

        self.discrete_action = discrete_action
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.batch_size = batch_size
        # self.alpha = alpha
        self.device = device
        self.actor = actor
        self.critic = critic
        self.buffer = replay_buffer
        self.max_action = torch.FloatTensor([max_action]).to(device)

        self.num_samples = vae_num_samples
        self.beta = beta
        self.lambd = alpha

        self.vae = VAE(state_dim=state_dim, action_dim=action_dim, latent_dim=2*action_dim, max_action=self.max_action, hidden_dim=vae_latent_dim)
        self.vae_optimizer = Adam(params=list(self.vae.parameters()), lr=vae_lr)
        self.n_iter = vae_train_iter
        if train_vae:
            self.vae_train(int(self.n_iter))

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
        

    def vae_train(self, n_iter):
        for i in range(n_iter):
            train_states, train_actions, _, _, _ = self.buffer.sample(batch_size=self.batch_size)

            # Variational Auto-Encoder Training
            recon, mean, std = self.vae(train_states, train_actions)

            recon_loss = F.mse_loss(recon, train_actions)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + self.beta * KL_loss
            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()

            if i % 10000 == 0:
                log.info("VAE training iteration {}, loss: {:.6f}".format(i, vae_loss.item()))


    def update_actor(self):
        state_batch, action_batch, _, _, _ = self.buffer.sample(batch_size=self.batch_size)
        # print(f"state {state_batch}, action {action_batch}")
        if state_batch is None:
            # Too few samples in the buffer to sample
            return
        Q, _, _ = self.get_min_q_target(state_batch, action_batch)
        neg_log_beta = self.vae.elbo_loss(state_batch, action_batch, self.beta, self.num_samples)
        actor_loss = - Q.mean() / Q.abs().mean().detach() + self.lambd * neg_log_beta.mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
        # print("pi", Q.size(), neg_log_beta.size())
        return actor_loss.detach().item()

    def update_critic(self):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.buffer.sample(batch_size=self.batch_size)
        with torch.no_grad():
            next_action, _, _ = self.actor.policy.sample(next_state_batch)
            next_q, _, _ = self.get_min_q_target(next_state_batch, next_action)
        q_target = reward_batch + (self.gamma * mask_batch * next_q)

        _, q1, q2 = self.get_min_q_value(state_batch, action_batch)
        critic1_loss = F.mse_loss(q_target, q1)
        critic2_loss = F.mse_loss(q_target, q2)
        q_loss = (critic1_loss + critic2_loss) * 0.5
        self.critic.optimizer.zero_grad()
        q_loss.backward()
        self.critic.optimizer.step()
        # print("q", q1.shape, q2.shape, q_target.shape, rewards.shape, dones.shape, actions.shape)
        return q_loss.detach().item(), critic1_loss.detach().item()
        

    def reset(self) -> None:
        pass