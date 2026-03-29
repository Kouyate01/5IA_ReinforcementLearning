import numpy as np
from typing import List, Tuple
import copy
from envs.base_env import BaseEnv


class GridWorld(BaseEnv):
    """
    Environnement GridWorld (2D) — grille 5x5.

    Le joueur démarre en haut à gauche (0, 0).
    L'objectif est d'atteindre la case en bas à droite (4, 4).
    Les bords bloquent le déplacement (le joueur reste en place).
    Chaque pas intermédiaire coûte -0.01.

    State encoding:
        Vecteur one-hot de taille rows*cols.
        Ex: grille 5x5, position (1,2) → index 7 vaut 1.0, reste 0.0.
    """

    ACTIONS = [0, 1, 2, 3]
    ACTION_NAMES = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}

    # (delta_row, delta_col) pour chaque action
    _MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, rows: int = 5, cols: int = 5):
        self._rows = rows
        self._cols = cols
        self._row = 0
        self._col = 0
        self._done = False
        self._last_reward = 0.0

    def reset(self) -> np.ndarray:
        self._row = 0
        self._col = 0
        self._done = False
        self._last_reward = 0.0
        return self.get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        assert not self._done, "Partie terminée, appelez reset()."
        assert action in self.ACTIONS, f"Action invalide : {action}"

        dr, dc = self._MOVES[action]
        new_row = self._row + dr
        new_col = self._col + dc

        # Si le déplacement sort de la grille, on reste en place
        if 0 <= new_row < self._rows and 0 <= new_col < self._cols:
            self._row = new_row
            self._col = new_col

        # Objectif atteint
        if self._row == self._rows - 1 and self._col == self._cols - 1:
            self._last_reward = 1.0
            self._done = True
        else:
            self._last_reward = -0.01

        return self.get_state(), self._last_reward, self._done

    def available_actions(self) -> List[int]:
        if self._done:
            return []
        return self.ACTIONS

    def is_game_over(self) -> bool:
        return self._done

    def get_state(self) -> np.ndarray:
        state = np.zeros(self._rows * self._cols, dtype=np.float32)
        state[self._row * self._cols + self._col] = 1.0
        return state

    def clone(self) -> "GridWorld":
        return copy.deepcopy(self)

    def render(self) -> None:
        print("+" + "---+" * self._cols)
        for r in range(self._rows):
            row_str = "|"
            for c in range(self._cols):
                if (r, c) == (self._row, self._col):
                    cell = " X "
                elif (r, c) == (self._rows - 1, self._cols - 1):
                    cell = " G "
                else:
                    cell = "   "
                row_str += cell + "|"
            print(row_str)
            print("+" + "---+" * self._cols)

    @property
    def state_size(self) -> int:
        return self._rows * self._cols

    @property
    def action_size(self) -> int:
        return len(self.ACTIONS)

    @property
    def current_player(self) -> int:
        return 0

    def num_players(self) -> int:
        return 1

    def score(self) -> float:
        return self._last_reward