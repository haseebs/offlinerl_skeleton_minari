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
from scipy import special
from copy import deepcopy

class FAC(BaseAgentDoubleQSingleV):
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
                 fname: str,
                 logq_entropic_index: float,
                 expq_entropic_index: float,
                 num_terms: int = 5,
                 ratio_eps: float = 10000,
                 use_exact: bool = False,
                 symmetric_coef: float = 1,
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

        self.behavior_policy = behavior_policy
        self.alpha = alpha
        self.fname = fname
        self.logq_entropic_index = logq_entropic_index
        self.expq_entropic_index = expq_entropic_index
        self.num_terms = int(num_terms)
        self.ratio_eps = ratio_eps
        self.symm_coef = symmetric_coef
        self.threshold = 1e-8
        self.exp_threshold = 10000

        # self.use_exact = use_exact
        self.use_exact = False
        # if (self.fname == "jensen_shannon" or self.fname == "gan") and self.use_exact:
        self.tawac_actor = deepcopy(actor)

        if self.num_terms == 2:
            self.num_terms += 1


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
            print(f"not enough samples, returning")
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

        v_value = self.get_v_value(state_batch)
        value_loss = F.mse_loss(current_min_q, v_value)
        self.critic.optimizer_v.zero_grad()
        value_loss.backward()
        self.critic.optimizer_v.step()


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
            # behavior_logprobs = self.behavior_policy.policy.log_prob(state_batch, action_batch)
        assert self.expq_entropic_index < 1.0, "q-exponential index should be less than 1 for filtering"

        """batch dim mean"""
        # baseline = q.mean(dim=0).unsqueeze(-1)
        # expq_adv = expq_x((min_q - baseline) * self.alpha, self.expq_entropic_index)
        expq_adv = expq_x((min_q - v) / self.alpha, self.expq_entropic_index)
        # expq_adv = torch.min(expq_adv, torch.FloatTensor([100.0]).to(state_batch.device))
        # expq_adv = torch.clip(expq_adv, min=self.threshold, max=self.exp_threshold)
        expq_adv = torch.clip(expq_adv,  max=self.exp_threshold)
        log_probs = self.actor.policy.log_prob(state_batch, action_batch)


        """tawac loss actor loss
        this policy is used for later Taylor expansion on policy ratio pi_learning / pi_tawac

        Modified on 2025/02/19:
        expand symmetric divergences into a TAWAC loss and a residual loss.
        Take Jeffrey for example, f_jeffrey(t) = t\lnt - \lnt
        D_{Jeffrey} = D_{KL}(\pi_{\theta} || \pi_{TKL}) + D_{KL}(\pi_{TKL} || \pi_{\theta})
        the second loss is TAWAC. The first loss is E_{\pi_{\theta}} [\ln \pi_{\theta} - \ln \pi_{TKL}]
        To avoid \ln \pi_{TKL} = \ln \pi_{ref} + \ln \exp_{q} (A) - \ln Z which can incur numerical issues,
        we use Taylor expansion to expand f(t) = - \lnt.

        To this end, we can first perform TAWAC update (seen as the second loss),
        followed by the Taylor expansion (residual loss, first term)
        """
        tawac_logprobs = self.tawac_actor.policy.log_prob(state_batch, action_batch)
        tawac_loss = -torch.mean(expq_adv * tawac_logprobs)
        self.tawac_actor.optimizer.zero_grad()
        tawac_loss.backward()
        self.tawac_actor.optimizer.step()


        # if self.use_exact:
        #     actor_loss = self.exact_fdiv(state_batch, expq_adv, log_probs, behavior_logprobs)
        # else:
        #     # make f(t) function TAWAC + residual
        tawac_loss, res_loss = self.Taylor_fdiv(state_batch, expq_adv, log_probs)
        # actor_loss = tawac_loss + self.symm_coef * res_loss
        actor_loss = tawac_loss + res_loss

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        """update the behavior policy
        f-divergence between learning policy and behavior policy
        """
        # behavior_logprobs = self.behavior_policy.policy.log_prob(state_batch, action_batch)
        # behavior_loss = -behavior_logprobs.mean()
        # self.behavior_policy.optimizer.zero_grad()
        # behavior_loss.backward()
        # self.behavior_policy.optimizer.step()


        return actor_loss.detach().item()
        # return tawac_loss.detach().item() + res_loss.detach().item()



    # def Taylor_fdiv(self, states, expq_adv, d_log_probs, num_terms=5):
    def Taylor_fdiv(self, states, expq_adv, d_log_probs):

        tawac_loss = -torch.mean(expq_adv * d_log_probs)

        sampled_actions, sampled_logprobs, _ = self.actor.policy.sample(states)
        with torch.no_grad():
            tawac_logprobs = self.tawac_actor.policy.log_prob(states, sampled_actions)
        res_ratio = self.clamp_ratio(torch.exp(tawac_logprobs - sampled_logprobs))


        # if self.num_terms < 2:
        #     return -torch.mean(expq_adv * logq_x(res_ratio, self.logq_entropic_index))


        if self.fname == "jeffrey":
            """
            D_{Jeffrey}(\pi_{\theta} || \pi_{TKL}) = E_{\pi_{\theta}}[f(\pi_{TKL} / \pi_{\theta})],
            t = \pi_{TKL} / \pi_{\theta}
            f(t) = t\lnt - \lnt = TAWAC - D_{KL}(\pi_{\theta} || \pi_{TKL})
            The latter term has \ln \pi_{TKL} which can be a numerical issue
            we Taylor expand the latter term, f(t) = -\lnt
            f^(n)(t) / n! = (-1)^n (n-1)! / (t^n n!)
            final loss: TAWAC + E_{\pi_{\theta}} [\pi_{\theta} / \pi_{TKL}]
            """
            res_series = torch.sum(torch.hstack([(-1)**q / q * self.clamp_ratio((res_ratio - 1)**q) for q in range(2, self.num_terms)]), dim=1, keepdim=True)

        elif self.fname == "jensen_shannon":
            """
            D_{JS}(\pi_{\theta} || \pi_{TKL}),
            f(t) =  t\lnt - (1+t)\ln 0.5*(1+t) = TAWAC - E_{\pi_{\theta}}[f^(n)(1)/n!]
            = TAWAC - E_{\pi_{\theta}}[(-1)^n (1 / n(n-1)2^(n-1)) \pi_{TKL} / \pi_{\theta}]
            """
            # series = torch.sum(torch.hstack([(-1)**q*(1 - 0.5**(q-2))/(q*(q-1)) * self.clamp_ratio((ratio - 1)**q) for q in range(2, num_terms)]), dim=1, keepdim=True)
            res_series = torch.sum(torch.hstack([(-1)**q / (2**(q-1) * q * (q-1)) * self.clamp_ratio((res_ratio - 1)**q) for q in range(2, self.num_terms)]), dim=1, keepdim=True)


        # elif self.fname == "gan":
        #     series = torch.sum(torch.hstack([(-1)**q*(1 - 0.5**(q-1))/(q*(q-1)) * self.clamp_ratio((ratio - 1)**q) for q in range(2, num_terms)]), dim=1, keepdim=True)


        else:
                raise NotImplementedError
        # return -torch.mean(expq_adv * res_series)
        # tawac loss + residual loss
        # residual loss Taylor expansion
        return tawac_loss, torch.mean(res_series)


    # def exact_fdiv(self, states, expq_adv, d_logprobs, d_behavior_logprobs):


    #     if self.fname == "jeffrey":
    #         """
    #         jeffrey = KL(expq_adv * behavior || learning) + KL(learning || expq_adv * behavior)
    #                = E_{behavior}[expq_adv * (\ln expq_adv*behavior - \ln learning )] + E_{learning}(\ln learning - \ln expq_adv - \ln behavior)
    #                = E_{dataset} [-expq_adv * \ln pi_learning] + E_{samples} [\ln pi_learning -  \ln behavior]


    #         Directly computing backward kl E_{samples} [\ln pi_learning -  \ln behavior]  could be numerically unstable
    #         use the unbiased estimator in http://joschu.net/blog/kl-approx.html
    #         KL(pi_1 || \pi_2) = E_{\pi_1} [ \pi_2 / \pi_1 + \ln\pi_1 - \ln\pi_2 - 1]

    #         Here, the policy being learned is \pi_1
    #         i.e. E_{\pi_\theta} [\pi_D / \pi_\theta + \ln\pi_\theta - \ln\pi_D - 1]
    #         """
    #         policy_actions, policy_logprobs, _ = self.actor.policy.sample(states)
    #         with torch.no_grad():
    #             behavior_logprobs = self.behavior_policy.policy.log_prob(states, policy_actions)

    #         # fwd_kl = torch.mean(expq_adv * (d_behavior_logprobs - d_logprobs))
    #         fwd_kl = -torch.mean(expq_adv * d_logprobs)

    #         # bwd_kl = torch.mean(policy_logprobs - behavior_logprobs)
    #         bwd_kl = torch.mean((self.clamp_ratio(torch.exp(behavior_logprobs - policy_logprobs)) - 1.) - (behavior_logprobs - policy_logprobs))
    #         # return fwd_kl + bwd_kl

    #     elif self.fname == "jensen_shannon":
    #         """js = 0.5*KL(expq_adv * behavior || 0.5*(exqp_adv*behavior + learning)) + 0.5*KL(learning || 0.5*(exqp_adv*behavior + learning))
    #         Remember in this case pi_{TKL} = expq_adv*behavior is not normalized. So this is really an approximation.
    #         But since the mixture anyway has 0.5*(pi_1 + pi_2), which is not separable using log. So it would be accurate when we fully know the two distributions

    #         TODO: can we have another layer of optimization that minimizes KL(pi_{TKL} || pi_{learning_1}), which is TAWAC.
    #         The overall policy is pi_{learning_2}, so now the JS divergence is:
    #         = 0.5 * E_{behavior}[expq_adv * (\ln expq_adv*behavior - log(0.5) - \ln (learning_1 + learning_2)] + 0.5 * E_{samples}[\ln learning_2 - log(0.5) - \ln (learning_1 + learning_2)]
    #         = 0.5 * E_{dataset}[-expq_adv * \ln (learning_1 + learning_2)] + 0.5 * E_{samples}[\ln learning_2 - log(0.5) - \ln (learning_1 + learning_2)]
    #         The optimization variable is learning_2.
    #         Though in general we cannot discard learning_1 in \ln (learning_1 + learning_2), because the log function is not separable,
    #         but we can easily sample from both distributions for optimization.

    #         In short, js_policy will take over the role of behavior policy.
    #         """
    #         # mixture = torch.log(0.5) + (d_behavior_logprobs + d_logprobs)
    #         # fwd_kl = 0.5*torch.mean(expq_adv * (d_behavior_logprobs - mixture))
    #         # bwd_kl = 0.5*torch.mean(policy_logprobs - mixture)
    #         sampled_actions, sampled_logprobs, _ = self.actor.policy.sample(states)
    #         with torch.no_grad():
    #             _, js_logprobs, _ = self.tawac_actor.sample(states)
    #             js_sampled_logprobs = self.tawac_actor.policy.log_prob(states, sampled_actions)

    #         fwd_kl = 0.5 * torch.mean(-expq_adv * torch.log(0.5*(js_logprobs.exp() + sampled_logprobs.exp())))
    #         log_mixture = torch.log(0.5*(js_sampled_logprobs.exp() + sampled_logprobs.exp()))
    #         # unbiased bwd estimation: KL(a||b) = E_{a} [b/a + \ln a - \ln b - 1]
    #         bwd_kl = 0.5 * torch.mean(self.clamp_ratio(torch.exp(log_mixture - sampled_logprobs)) - 1. - (log_mixture - sampled_logprobs))
    #         # bwd_kl =  0.5 * torch.mean(sampled_logprobs - torch.log(0.5) - torch.log(js_sampled_logprobs.exp() + sampled_logprobs.exp()))


    #     elif self.fname == "gan":
    #         """gan = E_{a} [\ln a - \ln (a+b)] + E_{b} [\ln b - \ln(a+b)]"""
    #         sampled_actions, sampled_logprobs, _ = self.actor.policy.sample(states)
    #         with torch.no_grad():
    #             _, js_logprobs, _ = self.tawac_actor.sample(states)
    #             js_sampled_logprobs = self.tawac_actor.policy.log_prob(states, sampled_actions)

    #         fwd_kl = torch.mean(-expq_adv * torch.log(js_logprobs.exp() + sampled_logprobs.exp()))
    #         log_mixture = torch.log(js_sampled_logprobs.exp() + sampled_logprobs.exp())
    #         # unbiased bwd estimation: KL(a||b) = E_{a} [b/a + \ln a - \ln b - 1]
    #         bwd_kl = torch.mean(self.clamp_ratio(torch.exp(log_mixture - sampled_logprobs)) - 1. - (log_mixture - sampled_logprobs))

    #     else:
    #         raise NotImplementedError(f"Unknown f divergence choice {self.fname}")

    #     return fwd_kl + bwd_kl




    # def clamp_ratio(self, ratio):
    #     return torch.clip(ratio, 0., 10000.)
    def clamp_ratio(self, ratio):
        # return torch.clip(ratio, 
        #                   max(0., 1-self.ratio_eps), 
        #                   1+self.ratio_eps)
        return torch.clip(ratio, 0., 10000.)




    def reset(self) -> None:
        return
