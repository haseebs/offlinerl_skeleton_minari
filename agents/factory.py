import gymnasium
from copy import deepcopy
from typing import Optional
from gymnasium.spaces.utils import flatdim
from .base_agent import BaseAgent
from .greedyac import GreedyAC

from .iql import IQL
from .tawac import TAWAC
from .fac import FAC
from .xql import XQL
from .sql import SQL
from .inac import InSampleAC
from .spot import SPOT

def get_agent(cfg: dict,
              discrete_action: bool,
              device: str,
              env: object,
              actor: object,
              critic: object,
              replay_buffer: object,

              ) -> BaseAgent:
    if isinstance(env, gymnasium.Env):
        state_dim = flatdim(env.observation_space)
        action_dim = flatdim(env.action_space)
        action_space = env.action_space
    else:
        raise NotImplementedError
    if cfg.name == "greedyac":
            proposal_actor = deepcopy(actor)
            return GreedyAC(discrete_action=discrete_action,
                            action_dim=action_dim,
                            state_dim=state_dim,
                            gamma=cfg.gamma,
                            batch_size=cfg.buffer.batch_size,
                            alpha=cfg.alpha,
                            device=device,
                            behavior_policy=actor,
                            proposal_policy=proposal_actor,
                            critic=critic,
                            replay_buffer=replay_buffer,
                            rho=cfg.rho,
                            n_action_proposals=cfg.n_action_proposals,
                            entropy_from_single_sample=cfg.entropy_from_single_sample)

    elif cfg.name == "iql":
            return IQL(discrete_action=discrete_action,
                            action_dim=action_dim,
                            state_dim=state_dim,
                            gamma=cfg.gamma,
                            batch_size=cfg.buffer.batch_size,
                            alpha=cfg.alpha,
                            device=device,
                            actor=actor,
                            critic=critic,
                            replay_buffer=replay_buffer,
                            expectile=cfg.expectile,
                            )
    elif cfg.name == "xql":
            return XQL(discrete_action=discrete_action,
                            action_dim=action_dim,
                            state_dim=state_dim,
                            gamma=cfg.gamma,
                            batch_size=cfg.buffer.batch_size,
                            alpha=cfg.alpha,
                            device=device,
                            actor=actor,
                            critic=critic,
                            replay_buffer=replay_buffer,
                            expectile=cfg.expectile,
                            sample_random_times=cfg.sample_random_times,
                            noise=cfg.noise,
                            vanilla=cfg.vanilla,
                            noise_std=cfg.noise_std,
                            log_loss=cfg.log_loss,
                            loss_temp=cfg.loss_temp,
                            )
    elif cfg.name == "sql":
            return SQL(discrete_action=discrete_action,
                            action_dim=action_dim,
                            state_dim=state_dim,
                            gamma=cfg.gamma,
                            batch_size=cfg.buffer.batch_size,
                            alpha=cfg.alpha,
                            device=device,
                            actor=actor,
                            critic=critic,
                            replay_buffer=replay_buffer,
                            )
    elif cfg.name == "tawac":
            return TAWAC(discrete_action=discrete_action,
                            action_dim=action_dim,
                            state_dim=state_dim,
                            gamma=cfg.gamma,
                            batch_size=cfg.buffer.batch_size,
                            alpha=cfg.alpha,
                            device=device,
                            actor=actor,
                            critic=critic,
                            replay_buffer=replay_buffer,
                            entropic_index=cfg.entropic_index,
                            )
    elif cfg.name == "inac":
            behavior_policy = deepcopy(actor)
            return InSampleAC(discrete_action=discrete_action,
                            action_dim=action_dim,
                            state_dim=state_dim,
                            gamma=cfg.gamma,
                            batch_size=cfg.buffer.batch_size,
                            alpha=cfg.alpha,
                            device=device,
                            actor=actor,
                            behavior_policy=behavior_policy,
                            critic=critic,
                            replay_buffer=replay_buffer,
                            )
    elif cfg.name == "spot":
            return SPOT(discrete_action=discrete_action,
                            action_dim=action_dim,
                            state_dim=state_dim,
                            gamma=cfg.gamma,
                            batch_size=cfg.buffer.batch_size,
                            alpha=cfg.alpha,
                            device=device,
                            actor=actor,
                            critic=critic,
                            replay_buffer=replay_buffer,
                            max_action=action_space.high,
                            beta=cfg.beta,
                            vae_num_samples=cfg.vae_num_samples,
                            vae_latent_dim=cfg.vae_latent_dim,
                            vae_optimizer=cfg.vae_optimizer,
                            vae_lr=cfg.vae_lr,
                            train_vae=cfg.train_vae,
                            vae_train_iter=cfg.vae_train_iter,
                            )
    elif cfg.name == "fac":
            behavior_policy = deepcopy(actor)
            return FAC(discrete_action=discrete_action,
                            action_dim=action_dim,
                            state_dim=state_dim,
                            gamma=cfg.gamma,
                            batch_size=cfg.buffer.batch_size,
                            alpha=cfg.alpha,
                            device=device,
                            actor=actor,
                            behavior_policy=behavior_policy,
                            critic=critic,
                            replay_buffer=replay_buffer,
                            logq_entropic_index=cfg.logq_entropic_index,
                            expq_entropic_index=cfg.expq_entropic_index,
                            fname=cfg.fname,
                            num_terms=cfg.num_terms,
                            ratio_eps=cfg.ratio_eps,
                            )

    elif cfg.name == "fac":
        behavior_policy = deepcopy(actor)
        return FAC(discrete_action=discrete_action,
                        action_dim=action_dim,
                        state_dim=state_dim,
                        gamma=cfg.gamma,
                        batch_size=cfg.buffer.batch_size,
                        alpha=cfg.alpha,
                        device=device,
                        actor=actor,
                        behavior_policy=behavior_policy,
                        critic=critic,
                        replay_buffer=replay_buffer,
                        logq_entropic_index=cfg.logq_entropic_index,
                        expq_entropic_index=cfg.expq_entropic_index,
                        fname=cfg.fname,
                        num_terms=cfg.num_terms,
                        ratio_eps=cfg.ratio_eps,
                        symmetric_coef=cfg.symmetric_coef,
                        )
    else:
            raise NotImplementedError



if __name__ == "__main__":
    pass
