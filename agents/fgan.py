from abc import ABC, abstractmethod
import logging
from jaxtyping import Float
import numpy as np
import torch
import torch.nn.functional as F
import os
import sys
from .base_agent import BaseAgentDoubleQSingleV
sys.path.append("../utils")
sys.path.append("../models")
from torch.optim import Adam
from utils.utils import logq_x, expq_x
from copy import deepcopy

log = logging.getLogger(__name__)

class FGAN(BaseAgentDoubleQSingleV):
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
                 fname: str,
                 T_lr: float,
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

        self.alpha = alpha
        self.entropic_index = entropic_index
        self.f_div = fname
        self.T_grad_steps = 1
        self.T_lr = T_lr  # 2025.06.02
        # self.T_lr 

        """
        Original f-GAN:
        T is a value function that is range-free. It takes state-action as input and outputs a scalar

        In RL we can consider another T candidate: log-policy, which is strictly non-positive
        The activation function needs to be changed correspondingly to the choice of T
        """
        self.T_type = "value"
        self.T = deepcopy(critic.q_value_net)
        # self.T_lr = critic.optimizer_q.param_groups[0]['lr']   # 2025.06.02 use the same lr as critic
        self.T_optimizer = Adam(params=list(self.T.parameters()), lr=self.T_lr)
        # self.gf_discriminator = self.gf_discriminator_value
        # self.gf_generator = self.gf_generator_value
        # if self.T_type == "log_prob":
        #     self.T = deepcopy(actor)
        #     self.T_optimizer = Adam(params=list(self.T.policy.parameters()), lr=T_lr)
        #     self.gf_discriminator = self.gf_discriminator_logprob
        #     self.gf_generator = self.gf_generator_logprob
        # elif self.T_type == "value":
        #     self.T = deepcopy(critic.q_value_net)
        #     self.T_optimizer = Adam(params=list(self.T.parameters()), lr=T_lr)
        #     self.gf_discriminator = self.gf_discriminator_value
        #     self.gf_generator = self.gf_generator_value
        # else:
        #     raise NotImplementedError


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


    def final_activation(self, x):
        if self.f_div == "jeffrey":
            # return -torch.exp(x)
            return -torch.exp(-x) # 2025.05.05
        elif self.f_div == "jensen_shannon":
            return x
        elif self.f_div == "gan":
            return x        
        else:
            raise NotImplementedError


    # def gf_discriminator_logprob(self, sampler_T, dataset_T, expq_adv):

    #     dataset_T = self.final_activation(dataset_T)
    #     # if self.f_div == "forward_kl":
    #     #     return -(torch.mean(expq_adv*torch.tan(0.5*torch.pi * torch.exp(dataset_T))) - torch.mean(torch.exp(sampler_T-1)))
    #     # elif self.f_div == "reverse_kl":
    #     #     return -(torch.mean(expq_adv*dataset_T) - torch.mean(-1-torch.log(-sampler_T)))
    #     if self.f_div == "jeffrey":
    #         # jeffrey expands into TAWAC + reverse KL variational representation
    #         return -(torch.mean(sampler_T) - torch.mean(expq_adv*(-1-torch.log(-dataset_T))))
    #     elif self.f_div == 'jensen_shannon':
    #         # return -(torch.mean(expq_adv*(torch.log(torch.tensor(2.))-torch.log(1-dataset_T))) - torch.mean(-torch.log(torch.tensor(2.)-2/(1-sampler_T))))
    #         # return -(torch.mean((torch.log(torch.tensor(2.))-torch.tensor(2.) - torch.log(1+torch.exp(-sampler_T)))) \
    #         #             - torch.mean(expq_adv*(torch.exp(-dataset_T))) )
    #         return -torch.mean(sampler_T) - torch.mean(expq_adv*(-dataset_T - 2*torch.exp(-dataset_T-1)))
    #     else:
    #         raise NotImplementedError

    # def gf_generator_logprob(self, dataset_T, expq_adv):
        
    #     dataset_T = self.final_activation(dataset_T)
    #     # if self.f_div == "forward_kl":
    #     #     return -torch.mean(torch.exp(sampler_T-1))
    #     # elif self.f_div == "reverse_kl":
    #     #     return -torch.mean(-1-torch.log(-sampler_T))
    #     if self.f_div == "jeffrey":
    #         return -torch.mean(expq_adv*(-1-torch.log(-dataset_T)))
    #     elif self.f_div == 'jensen_shannon':
    #         return -torch.mean(expq_adv*(torch.exp(-dataset_T)))
    #     else:
    #         raise NotImplementedError


    def gf_discriminator(self, sampler_T, dataset_T, expq_adv):
        
        dataset_gf = self.final_activation(dataset_T)
        sampler_gf = self.final_activation(sampler_T)
        if self.f_div == "jeffrey":
            # jeffrey expands into TAWAC + reverse KL variational representation
            # dataset_T = -torch.exp(dataset_T)
            # return -(torch.mean(-torch.exp(sampler_T)) - torch.mean(expq_adv*(-1-torch.log(-dataset_T))))
            # return -(torch.mean(expq_adv*dataset_gf) - torch.mean((-1-torch.log(-sampler_gf))))
            return -(torch.mean((-1-torch.log(-sampler_gf))) - torch.mean(expq_adv*dataset_gf) )  # equation 12
            # return -(torch.mean(expq_adv * (-1 - torch.log(-dataset_gf))) - torch.mean(sampler_gf)) # equation 11
        elif self.f_div == 'jensen_shannon':
            # return -(torch.mean((torch.log(torch.tensor(2.))-torch.tensor(2.) - torch.log(1+torch.exp(-sampler_T)))) \
            #             - torch.mean(expq_adv*(torch.exp(-dataset_T))) )
            # return -torch.mean(expq_adv*dataset_gf) - torch.mean((-sampler_gf - 2*torch.exp(-sampler_gf-1)))
            return -(torch.mean(-sampler_gf - 2*torch.exp(-sampler_gf-1)) - torch.mean(expq_adv*dataset_gf))  # equation 11
            # return -(torch.mean(expq_adv * (-dataset_gf - 2*torch.exp(-dataset_gf - 1))) - torch.mean(sampler_gf)) 

        elif self.f_div == "gan":
            return -(torch.mean(-sampler_gf - torch.exp(-sampler_gf-1)) - torch.mean(expq_adv*dataset_gf))  # equation 12
        else:
            raise NotImplementedError
        
    def gf_generator(self, sampler_T, expq_adv):

        sampler_gf = self.final_activation(sampler_T)
        # if self.f_div == "forward_kl":
        #     return -torch.mean(torch.exp(sampler_T-1))
        # elif self.f_div == "reverse_kl":
        #     return -torch.mean(-1-torch.log(-sampler_T))
        if self.f_div == "jeffrey":
            # return -torch.mean(expq_adv*(-1-torch.log(-dataset_T)))
            return -torch.mean(-1-torch.log(-sampler_gf)) # equation 12
            # return -torch.mean(sampler_gf) # equation 11
        elif self.f_div == 'jensen_shannon':
            # return -torch.mean(-torch.log(torch.tensor(2.)-2/(1-sampler_T)))
            return -torch.mean((-sampler_gf - 2*torch.exp(-sampler_gf-1))) # equation 11
            # return -torch.mean(sampler_gf) # equation 11
        elif self.f_div == 'gan':
            # return -torch.mean(-torch.log(torch.tensor(2.)-2/(1-sampler_T)))
            return -torch.mean((-sampler_gf - torch.exp(-sampler_gf-1))) # equation 12            
        else:
            raise NotImplementedError
    # def gf_generator(self, dataset_T, expq_adv):

    #     dataset_gf = self.final_activation(dataset_T)
    #     # if self.f_div == "forward_kl":
    #     #     return -torch.mean(torch.exp(sampler_T-1))
    #     # elif self.f_div == "reverse_kl":
    #     #     return -torch.mean(-1-torch.log(-sampler_T))
    #     if self.f_div == "jeffrey":
    #         # return -torch.mean(expq_adv*(-1-torch.log(-dataset_T)))
    #         return -torch.mean(expq_adv*(-1-torch.log(-dataset_T)))
    #     elif self.f_div == 'jensen_shannon':
    #         # return -torch.mean(-torch.log(torch.tensor(2.)-2/(1-sampler_T)))
    #         return -torch.mean(expq_adv*(torch.exp(-dataset_T)))
    #     else:
    #         raise NotImplementedError



    def _update_discriminator(self, state_batch, action_batch, expq_adv):
        """
        we use log prob as the T function
        """
        # with torch.no_grad():
        sampled_actions, _, _ = self.actor.policy.sample(state_batch)

        # if self.T_type == "log_prob":
        #     actor_values  = self.T.policy.log_prob(state_batch, sampled_actions)
        #     dataset_values = self.T.policy.log_prob(state_batch, action_batch)
        # elif self.T_type == "value":
        actor_values, _ = self.T(state_batch, sampled_actions)
        dataset_values, _ = self.T(state_batch, action_batch)

        T_loss = self.gf_discriminator(actor_values, dataset_values, expq_adv)

        self.T_optimizer.zero_grad()
        T_loss.backward()
        self.T_optimizer.step()

        return T_loss.detach().item()


    # def _update_generator(self, state_batch, action_batch, expq_adv):
    #     log_probs = self.actor.policy.log_prob(state_batch, action_batch)
    #     tawac_loss = -torch.mean(expq_adv * log_probs)

    #     # if self.T_type == "log_prob":
    #     #     generator_values = self.T.policy.log_prob(state_batch, action_batch)
    #     #     generator_values = torch.clip(generator_values, max=-1e-8)
    #     # elif self.T_type == "value":
    #     generator_values, _ = self.T(state_batch, action_batch)

    #     generator_loss =  self.gf_generator(generator_values, expq_adv)
    #     actor_loss = generator_loss + tawac_loss
    #     self.actor.optimizer.zero_grad()
    #     actor_loss.backward()
    #     self.actor.optimizer.step()
    #     return actor_loss.detach().item()
    def _update_generator(self, state_batch, action_batch, expq_adv):
        log_probs = self.actor.policy.log_prob(state_batch, action_batch)
        tawac_loss = -torch.mean(expq_adv * log_probs)

        # if self.T_type == "log_prob":
        #     generator_values = self.T.policy.log_prob(state_batch, action_batch)
        #     generator_values = torch.clip(generator_values, max=-1e-8)
        # elif self.T_type == "value":
        sampled_actions, _, _ = self.actor.policy.rsample(state_batch)
        # generator_values, _ = self.T(state_batch, action_batch)
        generator_values, _ = self.T(state_batch, sampled_actions)

        generator_loss =  self.gf_generator(generator_values, expq_adv)
        actor_loss = generator_loss + tawac_loss
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
        return actor_loss.detach().item()



    def update_actor(self):
        state_batch, action_batch, _, _, _ = self.buffer.sample(batch_size=self.batch_size)
        # print(f"state {state_batch}, action {action_batch}")
        if state_batch is None:
            # Too few samples in the buffer to sample
            return

        with torch.no_grad():
            min_q, _, _ = self.get_min_q_target(state_batch, action_batch)
            v = self.get_v_value(state_batch)
        expq_adv = expq_x((min_q - v) / self.alpha, self.entropic_index)

        T_loss = self._update_discriminator(state_batch, action_batch, expq_adv)
        # actor_loss = self._update_generator(state_batch, action_batch, expq_adv)
        actor_loss = self._update_generator(state_batch, action_batch, expq_adv)

        return actor_loss




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


    def reset(self) -> None:
        pass
