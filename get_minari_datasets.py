from itertools import product
import minari
import gymnasium

if __name__ == '__main__':
#    classes = ["mujoco"]
#    envs = ["hopper", "walker2d", "halfcheetah", "ant"]
#    levels = ["expert", "simple", "medium"]
#    versions = ["v0"]
    classes = ["D4RL"]
    envs = ["kitchen"]
    levels = ["partial", "complete", "mixed"]
    # envs = ["antmaze"]
    # levels = ["medium-play", "umaze-diverse", "large-diverse", "medium-diverse", "umaze", "large-play"]
    # versions = ["v1"]
    #    envs = ["pen", "hammer", "door", "relocate"]
    #    levels = ["expert", "cloned", "human"]
    versions = ["v2"]
   

    datasets = [lib + "/" + env + "/" + level + "-" +version for lib, env, level, version in product(classes, envs, levels, versions)]
    """check ~/.minari/datasets/ for datasets after running"""
    for dataset in datasets:
        print("downloading dataset:", dataset)
        _ = minari.load_dataset(dataset, download=True)
    #   env = gym.make(dataset)
    #   _ = env.get_dataset()


