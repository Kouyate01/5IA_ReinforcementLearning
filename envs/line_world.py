import numpy as np
from typing import List, Tuple
import copy
from base_env import BaseEnv


class LineWorld(BaseEnv):
    """
    Environnement LineWorld (1D).
    
    Le joueur se déplace sur une ligne de N cases.
    - Il commence à la position start_pos (milieu par défaut).
    - L'objectif est d'atteindre la case rightmost (case N-1) = récompense +1.
    - Tomber sur la case 0 = récompense -1.
    - Chaque pas intermédiaire = récompense 0.
    
    Actions:
        0 → aller à gauche
        1 → aller à droite
    
    State encoding:
        Vecteur one-hot de taille N (position courante).
        Ex: N=5, position=2 → [0, 0, 1, 0, 0]
    """

    def __init__(self, size: int = 5):
        """
        Args:
            size (int): Nombre de cases de la ligne (minimum 3).
        """
        assert size >= 3, "LineWorld doit avoir au moins 3 cases."
        self._size = size
        self._pos = size // 2
        self._done = False
        self._last_reward = 0.0

    # -------------------------------------------------------------------------
    # Implémentation des méthodes abstraites
    # -------------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        self._pos = self._size // 2
        self._done = False
        self._last_reward = 0.0
        return self.get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        assert not self._done, "La partie est terminée, appelez reset()."
        assert action in (0, 1), f"Action invalide : {action}"

        if action == 0:
            self._pos -= 1
        else:
            self._pos += 1

        if self._pos <= 0:
            self._pos = 0
            self._last_reward = -1.0
            self._done = True
        elif self._pos >= self._size - 1:
            self._pos = self._size - 1
            self._last_reward = 1.0
            self._done = True
        else:
            self._last_reward = 0.0

        return self.get_state(), self._last_reward, self._done

    def available_actions(self) -> List[int]:
        if self._done:
            return []
        return [0, 1]

    def is_game_over(self) -> bool:
        return self._done

    def get_state(self) -> np.ndarray:
        state = np.zeros(self._size, dtype=np.float32)
        state[self._pos] = 1.0
        return state

    def clone(self) -> "LineWorld":
        return copy.deepcopy(self)

    def render(self) -> None:
        line = ["."] * self._size
        line[self._pos] = "X"
        line[0] = "L"
        line[-1] = "R"
        print(f"[{''.join(line)}]  pos={self._pos}")

    @property
    def state_size(self) -> int:
        return self._size

    @property
    def action_size(self) -> int:
        return 2

    @property
    def current_player(self) -> int:
        return 0  # Environnement solo, toujours joueur 0

    def num_players(self) -> int:
        return 1

    def score(self) -> float:
        return self._last_reward