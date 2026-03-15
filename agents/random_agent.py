import numpy as np
import random
import time
from typing import List
from envs.base_env import BaseEnv


class RandomAgent:
    """
    Agent qui joue aléatoirement parmi les actions légales.
    Sert de baseline pour tous les autres agents.
    """

    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def select_action(self, env: BaseEnv) -> int:
        """
        Choisit une action aléatoire parmi les actions légales.
        
        Args:
            env: L'environnement courant.
            
        Returns:
            int: Index de l'action choisie.
        """
        actions = env.available_actions()
        assert len(actions) > 0, "Aucune action disponible !"
        return random.choice(actions)
