import minari
import logging
import numpy as np

log = logging.getLogger(__name__)


# https://github.com/Farama-Foundation/D4RL/blob/71a9549f2091accff93eeff68f1f3ab2c0e0a288/d4rl/gym_mujoco/__init__.py#L23-L31
HOPPER_RANDOM_SCORE = -20.272305
HALFCHEETAH_RANDOM_SCORE = -280.178953
WALKER_RANDOM_SCORE = 1.629008
ANT_RANDOM_SCORE = -325.6
ANTMAZE_RANDOM_SCORE = 0

HOPPER_EXPERT_SCORE = 3234.3
HALFCHEETAH_EXPERT_SCORE = 12135.0
WALKER_EXPERT_SCORE = 4592.3
ANT_EXPERT_SCORE = 3879.7
ANTMAZE_EXPERT_SCORE = 1


def get_normalized_score(env_name, minari_dataset, reward):
    return reward
    try:
        return minari.get_normalized_score(minari_dataset, reward)
    except:
        min_score = None
        max_score = None
        if 'hopper' in env_name.lower():
            min_score = HOPPER_RANDOM_SCORE
            max_score = HOPPER_EXPERT_SCORE
        elif 'halfcheetah' in env_name.lower():
            min_score = HALFCHEETAH_RANDOM_SCORE
            max_score = HALFCHEETAH_EXPERT_SCORE
        elif 'walker' in env_name.lower():
            min_score = WALKER_RANDOM_SCORE
            max_score = WALKER_EXPERT_SCORE
        elif 'antmaze' in env_name.lower():
            min_score = ANTMAZE_RANDOM_SCORE
            max_score = ANTMAZE_EXPERT_SCORE
        elif 'ant' in env_name.lower():
            min_score = ANT_RANDOM_SCORE
            max_score = ANT_EXPERT_SCORE
        else:
            raise NotImplementedError

        return (reward - min_score) / (max_score - min_score)


def evaluate(env, agent, episodes, seed, step):
    epsiode_rewards = []
    episode_lengths = []
    for e in range(episodes):
        obs = env.reset(seed=seed)
        length = 0
        while True:
            action = agent.act(obs, greedy=False)
            obs_next, reward, terminated, truncated, info = env.step(action)
            epsiode_rewards.append(reward)
            length += 1
            if terminated or truncated:
                obs = env.reset()
                episode_lengths.append(length)
                break
            obs = obs_next


def flatten_obs_dict(obs_dict):
    if isinstance(obs_dict, dict):
        return np.concatenate([
            obs_dict['observation'],
            obs_dict['achieved_goal'],
            obs_dict['desired_goal']
        ])
    return obs_dict.astype(np.float32)


def evaluate_episodic(env, agent, episodes, seed, step, env_name, minari_dataset):
    epsiode_rewards = []
    episode_lengths = []
    for e in range(episodes):
        obs = env.reset()[0]
        sum_rewards = 0
        length = 0
        while True:
            obs = flatten_obs_dict(obs)
            action = agent.act(obs, greedy=False)
            obs_next, reward, terminated, truncated, info = env.step(action)
            length += 1
            sum_rewards += reward
            if terminated or truncated:
                obs = env.reset(seed=seed)
                episode_lengths.append(length)
                epsiode_rewards.append(sum_rewards)
                break
            obs = obs_next

    """
    log performance as percentage of the original data collecting agent
    """
    log.info(f"Episode rewards: {epsiode_rewards}, lengths: {episode_lengths}")
    normalized = np.array([get_normalized_score(env_name, minari_dataset, ep_r) for ep_r in epsiode_rewards])
    mean_normalized = np.mean(normalized)
    max_normalized = np.max(normalized)
    min_normalized = np.min(normalized)
    median_normalized = np.median(normalized)
    std_normalized = np.std(normalized)
    mean_length = np.mean(episode_lengths)

    mean_reward = np.mean(epsiode_rewards)
    max_reward = np.max(epsiode_rewards)
    min_reward = np.min(epsiode_rewards)
    median_reward = np.median(epsiode_rewards)
    std_reward = np.std(epsiode_rewards)

    log.info(
        f"\nStep {step} evaluation:\n"
        f"  Normalized Rewards -> Mean: {mean_normalized:.2f}, Max: {max_normalized:.2f}, "
        f"Min: {min_normalized:.2f}, Median: {median_normalized:.2f}, Std: {std_normalized:.2f}\n"
        f"  Raw Rewards       -> Mean: {mean_reward:.2f}, Max: {max_reward:.2f}, "
        f"Min: {min_reward:.2f}, Median: {median_reward:.2f}, Std: {std_reward:.2f}\n"
        f"  Mean Episode Length: {mean_length}\n"
        f" ----------------------------------------------------- ")

    return mean_reward, std_reward, mean_normalized, std_normalized
