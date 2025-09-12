import numpy as np
import torch
from abc import ABC, abstractmethod
from jaxtyping import Float
from torchrl.data.datasets import MinariExperienceReplay
from tensordict import TensorDictBase


# Class definitions
class MinariBufferWrapper:
    def __init__(self,
                 dataset_id: str,
                 batch_size: int,
                 download: bool = True)-> None:
        self.buffer = MinariExperienceReplay(
                dataset_id=dataset_id,
                batch_size=batch_size,
                download=download
                )

    def sample(self, batch_size: int) -> tuple[list, list, list, list, list] :
        """
        Samples a random batch from the buffer

        Parameters
        ----------
        batch_size : int
            The size of the batch to sample

        Returns
        -------
        5-tuple of torch.Tensor
            The arrays of state, action, reward, next_state, and done from the
            batch
        """

        samples = self.buffer.sample(batch_size=batch_size)

        obs = samples['observation']
        if isinstance(obs, TensorDictBase):  # Nested observation
            state = torch.cat([v.float() for v in obs.values()], dim=-1)
        else:
            state = obs.float()

        next_obs = samples['next']['observation']
        if isinstance(next_obs, TensorDictBase):
            next_state = torch.cat([v.float() for v in next_obs.values()], dim=-1)
        else:
            next_state = next_obs.float()


        action = samples['action'].float()
        reward = samples['next']['reward'].float()
        done = 1 - torch.logical_or(samples['next']['terminated'],
                                    samples['next']['truncated']).float()


        return state, action, reward, next_state, done

    def __len__(self) -> int:
        """
        Gets the number of elements in the buffer

        Returns
        -------
        int
            The number of elements currently in the buffer
        """
        return len(self.buffer)




