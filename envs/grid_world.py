import numpy as np
from typing import List, Tuple
import copy
from base_env import BaseEnv


class GridWorld(BaseEnv):
    """
    Environnement GridWorld (2D).
    
    Le joueur se déplace sur une grille de rows x cols cases.
    - Il commence en haut à gauche (0, 0).
    - L'objectif est d'atteindre la case en bas à droite (rows-1, cols-1) = +1.
    - Des cases "trou" = récompense -1 et fin de partie.
    - Chaque pas = récompense -0.01 (encourage les chemins courts).
    - Les murs bloquent le déplacement (le joueur reste en place).
    
    Actions:
        0 → haut
        1 → bas
        2 → gauche
        3 → droite
    
    State encoding:
        Vecteur one-hot de taille rows*cols (position courante).
        Ex: grille 3x3, position (1,2) → index=5 → vecteur de taille 9
    """

    # Offsets (row, col) pour chaque action
    _MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    _ACTION_NAMES = ["↑", "↓", "←", "→"]

    def __init__(self, rows: int = 4, cols: int = 4, holes: List[Tuple[int, int]] = None):
        """
        Args:
            rows (int): Nombre de lignes.
            cols (int): Nombre de colonnes.
            holes (List[Tuple[int,int]]): Positions des cases "trou" (défaites).
        """
        assert rows >= 2 and cols >= 2, "Grille trop petite."
        self._rows = rows
        self._cols = cols
        
        # Trous par défaut si non spécifiés
        if holes is None:
            holes = [(1, 1), (2, 2)]  # trous par défaut pour 4x4
        self._holes = set(holes)
        
        # Vérifications
        assert (0, 0) not in self._holes, "La case de départ ne peut pas être un trou."
        assert (rows - 1, cols - 1) not in self._holes, "La case d'arrivée ne peut pas être un trou."

        self._row = 0
        self._col = 0
        self._done = False
        self._last_reward = 0.0

    # -------------------------------------------------------------------------
    # Implémentation des méthodes abstraites
    # -------------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        self._row = 0
        self._col = 0
        self._done = False
        self._last_reward = 0.0
        return self.get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        assert not self._done, "La partie est terminée, appelez reset()."
        assert 0 <= action <= 3, f"Action invalide : {action}"

        dr, dc = self._MOVES[action]
        new_row = self._row + dr
        new_col = self._col + dc

        # Mur → on reste en place
        if 0 <= new_row < self._rows and 0 <= new_col < self._cols:
            self._row = new_row
            self._col = new_col

        # Vérification de l'état
        if (self._row, self._col) in self._holes:
            self._last_reward = -1.0
            self._done = True
        elif self._row == self._rows - 1 and self._col == self._cols - 1:
            self._last_reward = 1.0
            self._done = True
        else:
            self._last_reward = -0.01  # pénalité de temps

        return self.get_state(), self._last_reward, self._done

    def available_actions(self) -> List[int]:
        if self._done:
            return []
        return [0, 1, 2, 3]

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
                elif (r, c) in self._holes:
                    cell = " O "
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
        return 4

    @property
    def current_player(self) -> int:
        return 0  # Solo

    def num_players(self) -> int:
        return 1

    def score(self) -> float:
        return self._last_reward