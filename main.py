import random
import hydra
import torch
import numpy as np
import logging

import os

from datetime import timedelta
from rich.pretty import pretty_repr
from timeit import default_timer as timer
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from utils import utils
from utils.evaluation import evaluate_episodic
from utils.utils import prep_cfg_for_db
from experiment import ExperimentManager, Metric
from models.actor import Actor
from models.critic import get_critic
from models.replay_buffers.minari_buffer_wrapper import MinariBufferWrapper
from agents.factory import get_agent
from environments.get_dataset import load_dataset, training_set_construction
from models.replay_buffers.factory import get_buffer
from gymnasium.wrappers import RecordVideo

import minari

log = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    start = timer()
    if HydraConfig.get().mode.value == 2:  # check whether its a sweep
        cfg.run += HydraConfig.get().job.num
        log.info(f'Running sweep... Run ID: {cfg.run}')
    log.info(f"Output directory  : \
             {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    cfg.agent.actor.optimizer.lr = cfg.agent.actor.optimizer.critic_lr_multiplier * \
                                    cfg.agent.critic.optimizer.lr
    flattened_cfg = prep_cfg_for_db(OmegaConf.to_container(cfg, resolve=True),
                                    to_remove=["schema", "db"])
    log.info(pretty_repr(flattened_cfg))

    exp = ExperimentManager(cfg.db_name, flattened_cfg, cfg.db_prefix, cfg.db)
    tables = {}
    for table_name in list(cfg.schema.keys()):
        columns = cfg.schema[table_name].columns
        primary_keys = cfg.schema[table_name].primary_keys
        tables[table_name] = Metric(table_name, columns, primary_keys, exp)
    torch.set_num_threads(cfg.n_threads)
    utils.set_seed(cfg.seed)



    dataset = minari.load_dataset(cfg.env)

    obs_list, act_list, r_list, next_obs_list, done_list = [], [], [], [],[]
    for ep in dataset.iterate_episodes():
        o = ep.observations
        a = ep.actions
        r = ep.rewards
        term = getattr(ep, "terminations", None)
        trunc = getattr(ep, "truncations", None)
        d = term | trunc
        next_o = o[1:]
        obs_list.append(o[:-1])
        act_list.append(a)
        r_list.append(r)
        next_obs_list.append(next_o)
        done_list.append(d)
    observations = np.concatenate(obs_list, axis=0).astype(np.float32)
    actions = np.concatenate(act_list, axis=0).astype(np.float32)
    rewards = np.concatenate(r_list, axis=0).astype(np.float32)
    next_observations = np.concatenate(next_obs_list, axis=0).astype(np.float32)
    dones = np.concatenate(done_list, axis=0).astype(np.float32)



    env = dataset.recover_environment()
    test_env = dataset.recover_environment(eval_env=True)
    env.reset(seed=cfg.seed)
    test_env.reset(seed=cfg.seed)

    buffer = get_buffer(cfg.agent.buffer, cfg.seed, env.env, cfg.device)
    for idx in range(len(observations)):
        """flip the terminating conditions so in the update only mask_batch needs to be used"""
        buffer.push(state=observations[idx],
                    action=actions[idx],
                    reward=rewards[idx],
                    next_state=next_observations[idx],
                    done=1-dones[idx])

    """
    test_env.unwrapped.render_mode = 'rgb_array'
    if cfg.run % 5 == 0:
        test_env = RecordVideo(test_env, video_folder=f"videos/{cfg.db_name}_{cfg.run}", name_prefix=env.spec.id,
                               #episode_trigger=lambda x: True)
                               episode_trigger=lambda x: x % 2 == 0)
                               #episode_trigger=lambda x: x % cfg.evaluation_episodes == 0)
    """

    actor = Actor(cfg.agent.actor, env, cfg.agent.store_old_policy, cfg.device)
    critic = get_critic(cfg.agent.critic, env, cfg.device)
    agent = get_agent(cfg.agent, False, cfg.device, env, actor, critic, buffer)

    """
    we sample 10 states for evaluating policy parameters like location and scale
    parameters from algorithms with the same seed are compared, e.g. seed=0
    """
    # policy_param_states, _, _, _, _ = buffer.sample(batch_size=10)

    step = 0
    all_rewards = []
    all_normalized_rewards = []
    episode = 0

    for step in range(cfg.steps):
        """learn from offline datasets"""
        losses = agent.update_critic()
        actor_loss = agent.update_actor()
        q_loss, v_loss = losses

        if step % cfg.evaluation_steps == 0:
            log.info(f"Step {step}, \t training actor loss: {actor_loss:.4f}, \t critic q loss: {q_loss:.4f} v loss: {v_loss:.4f}")
            episode += 0
            mean_reward, std_reward, mean_normalized, std_normalized = evaluate_episodic(
                    test_env,
                    agent,
                    cfg.evaluation_episodes,
                    cfg.seed,
                    step,
                    cfg.env,
                    dataset,
                    )
            tables["returns"].add_data(
                [
                    cfg.run,
                    step,
                    episode,
                    mean_reward,
                    mean_normalized,
                ]
            )
            all_rewards.append(mean_reward)
            all_normalized_rewards.append(mean_normalized)

        if step % 10000 == 0:
            tables["returns"].commit_to_database()
            # tables["policy"].commit_to_database()

    total_time = timedelta(seconds=timer() - start).seconds / 60
    auc_10 = float(np.mean(all_rewards[-int(len(all_rewards)*0.1):]))
    auc_50 = float(np.mean(all_rewards[-int(len(all_rewards)*0.5):]))
    auc_100 = float(np.mean(all_rewards))
    norm_auc_10 = float(np.mean(all_normalized_rewards[-int(len(all_normalized_rewards)*0.1):]))
    norm_auc_50 = float(np.mean(all_normalized_rewards[-int(len(all_normalized_rewards)*0.5):]))
    norm_auc_100 = float(np.mean(all_normalized_rewards))
    tables["summary"].add_data(
        [
            cfg.run,
            step,
            episode,
            auc_100,
            norm_auc_100,
            auc_50,
            norm_auc_50,
            auc_10,
            norm_auc_10,
            total_time
        ]
    )
    tables["returns"].commit_to_database()
    tables["summary"].commit_to_database()
    tables["policy"].commit_to_database()
    log.info(f'Total time taken: {total_time}  minutes')

    actor.save(f"saved/iql_{cfg.seed}.pt")

if __name__ == "__main__":
    main()
