from .simulations import SimEnv3

def get_env(cfg: dict, seed: int):
    if cfg.name == "SimEnv3":
        return SimEnv3(seed=seed)
    else:
        raise NotImplementedError
