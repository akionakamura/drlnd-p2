from dataclasses import dataclass
from typing import List

import numpy as np



@dataclass(frozen=True, eq=True)
class RunConfig:
    """Holds the main configurations of a run, to facilitate experimentation."""
    learn_step: int
    sync_step: int
    batch_size: int
    gamma: float
    epsilon_decay: float


@dataclass
class RunExperiments:
    """Holds the several configurations to run experiments to."""
    learn_steps: List[int]
    sync_steps: List[int]
    batch_sizes: List[int]
    gammas: List[float]
    epsilon_decays: List[float]

    def get_random(self) -> RunConfig:
        """Gets a random setup."""

        return RunConfig(
            learn_step=np.random.choice(self.learn_steps, 1)[0],
            sync_step=np.random.choice(self.sync_steps, 1)[0],
            batch_size=np.random.choice(self.batch_sizes, 1)[0],
            gamma=np.random.choice(self.gammas, 1)[0],
            epsilon_decay=np.random.choice(self.epsilon_decays, 1)[0],
        )

    def get_configs(self, n_configs):
        combinations = np.array(np.meshgrid(
            self.learn_steps,
            self.sync_steps,
            self.batch_sizes,
            self.gammas,
            self.epsilon_decays
        )).T.reshape(-1, 5)

        n = min(n_configs, combinations.shape[0])
        idxs = np.random.choice(combinations.shape[0], n)

        return [RunConfig(
            learn_step=combinations[i, 0],
            sync_step=combinations[i, 1],
            batch_size=combinations[i, 2],
            gamma=combinations[i, 3],
            epsilon_decay=combinations[i, 4],
        ) for i in idxs]