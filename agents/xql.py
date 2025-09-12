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



class XQL(BaseAgentDoubleQSingleV):
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
                 sample_random_times: int = 1,
                 noise: bool = True,
                 vanilla: bool = True,
                 noise_std: float = 0.1,
                 log_loss: bool = True,
                 loss_temp: float = 1.0,
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

        self.sample_random_times = sample_random_times
        self.noise = noise
        self.vanilla = vanilla
        self.noise_std = noise_std
        self.log_loss = log_loss
        self.loss_temp = loss_temp

        self.expectile = expectile
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

        if self.sample_random_times > 0:
            # add random action_batch to smooth loss computation (use 1/2(rho + Unif))
            times = self.sample_random_times
            # random_action = jax.random.uniform(
            #     rng1, shape=(times * action_batch.shape[0],
            #                  action_batch.shape[1]),
            #     minval=-1.0, maxval=1.0)
            random_action = torch.rand(times * action_batch.shape[0], action_batch.shape[1]) * 2.0 - 1.0
            obs = torch.concatenate([state_batch for _ in range(self.sample_random_times + 1)], axis=0)
            acts = torch.concatenate([action_batch, random_action], axis=0)
        else:
            obs = state_batch
            acts = action_batch

        if self.noise:
            std = self.noise_std
            # noise = jax.random.normal(rng2, shape=(acts.shape[0], acts.shape[1]))
            noise = torch.normal(mean=torch.zeros(acts.shape), std=torch.ones(acts.shape))
            noise = torch.clip(noise * std, -0.5, 0.5)
            acts = (acts + noise)
            acts = torch.clip(acts, -1, 1)

        q, _, _ = self.get_min_q_value(obs, acts)

        # v = value.apply({'params': value_params}, obs)
        v = self.get_v_value(obs)

        """update value network"""
        if self.vanilla:
            v_loss = self.expectile_loss(q - v, self.expectile).mean()
        else:
            if self.log_loss:
                v_loss = self.gumbel_log_loss(q - v, alpha=self.loss_temp).mean()
            else:
                v_loss = self.gumbel_rescale_loss(q - v, alpha=self.loss_temp).mean()

        self.critic.optimizer_v.zero_grad()
        v_loss.backward()
        self.critic.optimizer_v.step()

        """update q network"""
        with torch.no_grad():
            next_v = self.get_v_value(next_state_batch)
            v = self.get_v_value(state_batch)

        target_q = reward_batch + self.gamma * mask_batch * next_v

        # q1, q2 = critic.apply({'params': critic_params}, batch.observations, acts)
        # v = target_value(batch.observations)
        _, q1, q2 = self.get_min_q_value(state_batch, action_batch)


        def mse_loss(q, q_target, *args):
            x = q - q_target
            loss = self.huber_loss(x, delta=20.0)  # x**2
            return loss.mean()

        critic_loss = mse_loss

        loss1 = critic_loss(q1, target_q, v, self.loss_temp)
        loss2 = critic_loss(q2, target_q, v, self.loss_temp)
        q_loss = (loss1 + loss2).mean()

        self.critic.optimizer_q.zero_grad()
        q_loss.backward()
        self.critic.optimizer_q.step()
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
        exp_adv = torch.exp((q - v) / self.alpha)
        # exp_adv = torch.min(exp_adv, torch.FloatTensor([100.0]).to(state_batch.device))
        exp_adv = torch.clip(exp_adv, 0., 100.0)
        log_probs = self.actor.policy.log_prob(state_batch, action_batch)
        assert exp_adv.shape == log_probs.shape, f"exp_adv shape {exp_adv.shape}, log_probs shape {log_probs.shape}"
        actor_loss = -(exp_adv * log_probs).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
        return actor_loss.detach().item()


    def huber_loss(self, x, delta: float = 1.):
        """Huber loss, similar to L2 loss close to zero, L1 loss away from zero.
        See "Robust Estimation of a Location Parameter" by Huber.
        (https://projecteuclid.org/download/pdf_1/euclid.aoms/1177703732).
        Args:
        x: a vector of arbitrary shape.
        delta: the bounds for the huber loss transformation, defaults at 1.
        Note `grad(huber_loss(x))` is equivalent to `grad(0.5 * clip_gradient(x)**2)`.
        Returns:
        a vector of same shape of `x`.
        """
        # 0.5 * x^2                  if |x| <= d
        # 0.5 * d^2 + d * (|x| - d)  if |x| > d
        abs_x = torch.abs(x)
        quadratic = torch.clip(abs_x, 0, delta)
        # Same as max(abs_x - delta, 0) but avoids potentially doubling gradient.
        linear = abs_x - quadratic
        # print("huber", abs_x.size(), quadratic.size())
        return 0.5 * quadratic ** 2 + delta * linear

    def gumbel_log_loss(self, diff, alpha=1.0):
        """ Gumbel loss J: E[e^x - x - 1]. We can calculate the log of Gumbel loss for stability, i.e. Log(J + 1)
        log_gumbel_loss: log((e^x - x - 1).mean() + 1)
        """
        diff = diff
        x = diff / alpha
        grad = self.grad_gumbel(x, alpha)
        # use analytic gradients to improve stability
        # loss = jax.lax.stop_gradient(grad) * x
        loss = grad.detach() * x
        # print("gumbel log", grad.size(), diff.size(), x.size())
        return loss

    def grad_gumbel(self, x, alpha, clip_max=7):
        """Calculate grads of log gumbel_loss: (e^x - 1)/[(e^x - x - 1).mean() + 1]
        We add e^-a to both numerator and denominator to get: (e^(x-a) - e^(-a))/[(e^(x-a) - xe^(-a)).mean()]
        """
        # clip inputs to grad in [-10, 10] to improve stability (gradient clipping)
        x = torch.clip(x, -np.inf, clip_max)  # jnp.clip(x, a_min=-10, a_max=10)

        # calculate an offset `a` to prevent overflow issues
        x_max = torch.max(x, axis=0)[0]
        # choose `a` as max(x_max, -1) as its possible for x_max to be very small and we want the offset to be reasonable
        x_max = torch.where(x_max < -1, -1, x_max)
        # keep track of original x
        x_orig = x
        # offsetted x
        x1 = x - x_max

        grad = (torch.exp(x1) - torch.exp(-x_max)) / \
               (torch.mean(torch.exp(x1) - x_orig * torch.exp(-x_max), axis=0, keepdims=True))
        # print("gumbel", x1.size(), x_max, x_orig.size())
        return grad

    def reset(self) -> None:
        return
