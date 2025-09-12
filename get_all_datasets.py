from itertools import product
import d4rl
import gym

if __name__ == '__main__':
   envs = ["hopper", "walker2d", "halfcheetah", "ant"]
   levels = ["expert", "medium-expert", "medium", "medium-replay"]
   versions = ["v2"]
   
   """Also possible to see all registered d4rl datasets here:

    # Print all registered environments in gym
    envs = gym.envs.registry.all()
    d4rl_envs = [env_spec.id for env_spec in envs]

    print("Available D4RL datasets:")
    print("\n".join(d4rl_envs))
   """

   datasets = [env + "-" + level + "-" + version for env, level, version in product(envs, levels, versions)]
   """check ~/.d4rl for datasets after running"""
   for dataset in datasets:
      env = gym.make(dataset)
      _ = env.get_dataset()


