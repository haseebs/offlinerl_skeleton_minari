from abc import ABC, abstractmethod
from jaxtyping import Float
import numpy as np
import torch
from models.policy_parameterizations.student import Student

class BaseAgent(ABC):
    """
    Class BaseAgent implements the base functionality for all agents
    """
    def __init__(self):
        pass

    @abstractmethod
    def act(self, state: Float[np.ndarray, "state_dim"]) -> Float[np.ndarray, "action_dim"]:
        pass

    @abstractmethod
    def update_critic(self) -> float:
        pass

    @abstractmethod
    def update_actor(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    def get_policy_params(self, states) -> None:
        states = torch.FloatTensor(states).to(self.device)
        dist_params = self.actor.policy(states)

        if isinstance(self.actor.policy, Student):
            loc, scale, dof = dist_params
        else:
            loc, scale = dist_params
            dof = torch.FloatTensor([np.inf])
        
        loc = loc.detach().cpu().numpy()
        scale = scale.detach().cpu().numpy()
        return loc, scale, dof


class BaseAgentDoubleQ(BaseAgent):
    """
    Base Agent that has two Q networks
    """

    def get_min_q_value(self, state_batch, action_batch):
        q1, q2 = self.critic.q_value_net(state_batch, action_batch)
        return torch.min(q1, q2), q1, q2

    def get_min_q_target(self, state_batch, action_batch):
        q1, q2 = self.critic.q_target_net(state_batch, action_batch)
        return torch.min(q1, q2), q1, q2   


class BaseAgentDoubleQSingleV(BaseAgentDoubleQ):
    
    def get_v_value(self, state_batch):
        return self.critic.v_value_net(state_batch)