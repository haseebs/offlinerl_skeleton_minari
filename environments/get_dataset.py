import pickle
import time
import copy
import numpy as np
import os

import gymnasium
from .factory import get_env


def training_set_construction(data_dict):

    assert len(list(data_dict.keys())) == 1
    data_dict = data_dict[list(data_dict.keys())[0]]
    states = data_dict['states']
    actions = data_dict['actions']
    rewards = data_dict['rewards']
    next_states = data_dict['next_states']
    terminations = data_dict['terminations']

    return [states, actions, rewards, next_states, terminations]


def load_dataset(env_name, dataset, cfg, seed):
    print(env_name, dataset)
    path = None

    if env_name == 'SimEnv3':
        path = {"generate": None}

    assert path is not None

    datasets = {}
    for name in path:
        if name == "env":
            env = gymnasium.make(path['env'])
            try:
                data = env.get_dataset()
            except:
                env = env.unwrapped
                data = env.get_dataset()
            datasets[name] = {
                'states': data['observations'],
                'actions': data['actions'],
                'rewards': data['rewards'],
                'next_states': data['next_observations'],
                'terminations': data['terminals'],
            }
        elif name == "generate":
            env = get_env(cfg.env, seed)
            datasets[name] = env.get_dataset()
        else:
            raise NotImplementedError
        return datasets
    else:
        return {}
