# envs/bobail_vs_random.py
import random
import numpy as np
from typing import List, Tuple, Optional
from envs.bobail import Bobail


class BobailVsRandom(Bobail):
    """
    Wrapper autour de Bobail où l'adversaire (joueur 1 par défaut)
    joue des coups aléatoires de façon transparente.

    L'agent entraîné joue toujours en tant que `agent_player` (0 par défaut).
    Quand c'est au tour de l'adversaire, le wrapper enchaîne automatiquement
    ses coups (phase 1 ET phase 2) avant de rendre la main à l'agent.

    Usage :
        env = BobailVsRandom(agent_player=0)
        state = env.reset()
        while not env.is_game_over():
            action = agent.choose_action(state, env.available_actions())
            state, reward, done = env.step(action)
    """

    def __init__(self, agent_player: int = 0):
        super().__init__()
        assert agent_player in (0, 1), "agent_player doit être 0 ou 1"
        self.agent_player = agent_player

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        state = super().reset()
        # Si l'adversaire commence (agent_player == 1), on joue ses coups
        state = self._play_random_if_opponent_turn(state)
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        L'agent joue `action`, puis l'adversaire joue automatiquement
        (ses deux phases s'il le faut) avant de rendre la main.
        """
        assert not self._done, "Partie terminée, appelez reset()."
        assert self._current_player == self.agent_player, (
            "Ce n'est pas le tour de l'agent !"
        )

        # --- Coup de l'agent ---
        state, reward, done = super().step(action)
        if done:
            return state, self._agent_reward(), done

        # --- Tour(s) complet(s) de l'adversaire ---
        state = self._play_random_if_opponent_turn(state)

        # La récompense finale correspond au point de vue de l'agent
        if self._done:
            reward = self._agent_reward()

        return state, reward, self._done

    # ------------------------------------------------------------------
    # Interne
    # ------------------------------------------------------------------

    def _play_random_if_opponent_turn(self, state: np.ndarray) -> np.ndarray:
        """
        Joue des coups random tant que c'est le tour de l'adversaire
        (gère les deux phases du tour adverse).
        """
        while not self._done and self._current_player != self.agent_player:
            actions = self.available_actions()
            if not actions:
                break
            action = random.choice(actions)
            state, _, _ = super().step(action)
        return state

    def _agent_reward(self) -> float:
        """Récompense du point de vue de l'agent."""
        if self._winner is None:
            return 0.0
        return 1.0 if self._winner == self.agent_player else -1.0

    # Dans BobailVsRandom, surcharge score() pour le point de vue de l'agent
    def score(self) -> float:
        if self._winner is None:
            return 0.5
        return 1.0 if self._winner == self.agent_player else 0.0