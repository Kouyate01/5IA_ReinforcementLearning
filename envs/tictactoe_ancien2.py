import numpy as np
from typing import List, Tuple, Optional
import copy
from envs.base_env import BaseEnv


class TicTacToe(BaseEnv):
    """
    Environnement TicTacToe (Morpion) 3x3.

    Joueurs :
        - Joueur 0 : X
        - Joueur 1 : O

    Actions :
        0-8 -> cases de la grille de gauche à droite, haut en bas

    Récompenses :
        +1.0 -> victoire
        -1.0 -> défaite
         0.0 -> nul / coup intermédiaire
        (Le reward est donné du point de vue du joueur qui vient de jouer.)

    État :
        Vecteur de taille 27 :
            - 0..8   : cases occupées par le joueur 0
            - 9..17  : cases occupées par le joueur 1
            - 18..26 : cases vides
    """

    _WINNING_COMBOS = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6),
    ]

    def __init__(self):
        self._board = np.zeros(9, dtype=np.int8)
        self._current_player = 0
        self._done = False
        self._winner: Optional[int] = None

    def reset(self) -> np.ndarray:
        self._board = np.zeros(9, dtype=np.int8)
        self._current_player = 0
        self._done = False
        self._winner = None
        return self.get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert not self._done, "La partie est terminée, appelez reset()."
        assert action in self.available_actions(), f"Action illégale : {action}"

        player = self._current_player
        self._board[action] = player + 1

        reward = 0.0

        if self._check_winner(player):
            self._done = True
            self._winner = player
            reward = 1.0
        elif np.all(self._board != 0):
            self._done = True
            self._winner = None
            reward = 0.0

        if not self._done:
            self._current_player = 1 - self._current_player

        return self.get_state(), reward, self._done

    def available_actions(self) -> List[int]:
        if self._done:
            return []
        return [i for i in range(9) if self._board[i] == 0]

    def is_game_over(self) -> bool:
        return self._done

    def get_state_ancien(self) -> np.ndarray:
        state = np.zeros(27, dtype=np.float32)
        for i in range(9):
            if self._board[i] == 1:
                state[i] = 1.0
            elif self._board[i] == 2:
                state[9 + i] = 1.0
            else:
                state[18 + i] = 1.0
        return state

    def get_state(self) -> np.ndarray:
        """
        État du point de vue du joueur courant :
        - ses coups -> couche joueur 0
        - coups adverses -> couche joueur 1
        - cases vides -> couche vide
        """
        state = np.zeros(27, dtype=np.float32)
        me = self._current_player + 1
        opp = 2 if me == 1 else 1

        for i in range(9):
            if self._board[i] == me:
                state[i] = 1.0
            elif self._board[i] == opp:
                state[9 + i] = 1.0
            else:
                state[18 + i] = 1.0
        return state

    def clone(self) -> "TicTacToe":
        return copy.deepcopy(self)

    def render(self) -> None:
        symbols = {0: ".", 1: "X", 2: "O"}
        print()
        for row in range(3):
            cells = [symbols[self._board[row * 3 + col]] for col in range(3)]
            print(f" {cells[0]} | {cells[1]} | {cells[2]} ")
            if row < 2:
                print("---+---+---")
        if self._done:
            if self._winner is not None:
                print(f"\n→ Joueur {self._winner} ({'X' if self._winner == 0 else 'O'}) gagne !")
            else:
                print("\n→ Match nul !")
        else:
            print(f"\n→ Tour du joueur {self._current_player} ({'X' if self._current_player == 0 else 'O'})")
        print()

    @property
    def state_size(self) -> int:
        return 27

    @property
    def action_size(self) -> int:
        return 9

    @property
    def current_player(self) -> int:
        return self._current_player

    def score(self) -> float:
        """
        Score final pour le joueur 0 :
            1.0 si victoire
            0.5 si nul
            0.0 si défaite
        """
        if not self._done:
            raise ValueError("Le score n'est disponible qu'en fin de partie.")
        if self._winner == 0:
            return 1.0
        if self._winner is None:
            return 0.5
        return 0.0

    def _check_winner(self, player: int) -> bool:
        val = player + 1
        for combo in self._WINNING_COMBOS:
            if all(self._board[i] == val for i in combo):
                return True
        return False