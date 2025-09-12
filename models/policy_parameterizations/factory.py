import torch.nn as nn
import gymnasium
from gymnasium.spaces.utils import flatdim
from .gaussian import Gaussian
from .softmax import Softmax
from .squashed_gaussian import SquashedGaussian
from .beta import Beta
from .student import Student
from .q_gaussian import qMultivariateGaussian, qHeavyTailedGaussian

def get_policy(cfg: dict, env: object, device: str) -> nn.Module:

        if isinstance(env, gymnasium.Env):
                num_inputs = flatdim(env.observation_space)
                num_actions = flatdim(env.action_space)
                action_space = env.action_space
        else:
                raise NotImplementedError


        if cfg.policy == "gaussian":
                return Gaussian(num_inputs,
                                num_actions,
                                cfg.hidden_dim,
                                cfg.activation,
                                action_space,
                                cfg.clip_stddev,
                                cfg.init).to(device)
        elif cfg.policy == "squashed_gaussian":
                return SquashedGaussian(num_inputs,
                                        num_actions,
                                        cfg.hidden_dim,
                                        cfg.activation,
                                        action_space,
                                        cfg.clip_stddev,
                                        cfg.init).to(device)
        elif cfg.policy == "beta":
                return Beta(num_inputs,
                                num_actions,
                                cfg.hidden_dim,
                                cfg.activation,
                                action_space,
                                cfg.init).to(device)
        elif cfg.policy == "student":
                return Student(num_inputs,
                                num_actions,
                                cfg.hidden_dim,
                                cfg.activation,
                                action_space,
                                cfg.init).to(device)
        elif cfg.policy == "q_gaussian":
                return qMultivariateGaussian(
                                num_inputs=num_inputs,
                                num_actions=num_actions,
                                hidden_dim=cfg.hidden_dim,
                                activation=cfg.activation,
                                action_space=action_space,
                                entropic_index=cfg.entropic_index,
                                init=cfg.init).to(device)
        elif cfg.policy == "heavytailed_gaussian":
                return qHeavyTailedGaussian(
                                num_inputs=num_inputs,
                                num_actions=num_actions,
                                hidden_dim=cfg.hidden_dim,
                                activation=cfg.activation,
                                action_space=action_space,
                                entropic_index=cfg.entropic_index,
                                init=cfg.init).to(device)
        else:
                raise NotImplementedError
