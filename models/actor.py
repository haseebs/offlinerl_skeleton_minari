import torch
import torch.nn as nn
from gymnasium import Env
from jaxtyping import Float
from .optimizer_factory import get_optimizer
from .policy_parameterizations.factory import get_policy
from .critic import get_critic
import copy


class Actor:
    def __init__(self, cfg: dict, env: Env, store_old_policy: bool = False, device: str = 'cpu') -> None:
        """
        cfg contains:
            - policy parameters
            - optimizer params
        env: Environment to infer action and state shapes from
        store_old_policy: see self.policy_backup_hook() for details
        """
        self.policy = get_policy(cfg.policy, env, device)
        self.policy_old = None
        self.optimizer = get_optimizer(cfg.optimizer, list(self.policy.parameters()))
        self.cfg = cfg
        if store_old_policy:
            self.policy_old = copy.deepcopy(self.policy)
            self.hook = self.optimizer.register_step_pre_hook(self.policy_backup_hook)

    def policy_backup_hook(self, *args) -> None:
        """
        store the policy's parameters in self.policy_old whenever we call
        a self.optimzier.step(). This is so we will always have a set of
        previous policy parameters available for use in KL computations
        """
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.zero_grad()

    def sample(self, state: Float[torch.Tensor, "state_dim"]) -> Float[torch.Tensor, "action_dim"]:
        return self.policy.sample(state)

    def save(self, path: str) -> None:
        checkpoint = {
            "policy_state": self.policy.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "cfg": self.cfg,
        }
        if self.policy_old is not None:
            checkpoint["policy_old_state"] = self.policy_old.state_dict()

        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, env: Env, device: str = "cpu") -> "Actor":
        checkpoint = torch.load(path, map_location=device,
                                weights_only=False)

        # reconstruct actor with saved cfg
        actor = cls(checkpoint["cfg"], env, device=device,
                    store_old_policy="policy_old_state" in checkpoint)

        # load weights
        actor.policy.load_state_dict(checkpoint["policy_state"])
        actor.optimizer.load_state_dict(checkpoint["optimizer_state"])

        if "policy_old_state" in checkpoint and actor.policy_old is not None:
            actor.policy_old.load_state_dict(checkpoint["policy_old_state"])

        return actor

if __name__ == "__main__":
    pass

