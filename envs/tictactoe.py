# envs/tictactoe.py

import numpy as np
import copy
from typing import List, Tuple, Optional
from envs.base_env import BaseEnv
from agents.random_agent import RandomAgent


class TicTacToe(BaseEnv):
    """
    Environnement TicTacToe (Morpion) 3x3 — joueur 0 (X) vs Random (O).

    L'agent est toujours le joueur 0 (X) et commence toujours.
    Après chaque coup de l'agent, l'adversaire random joue immédiatement.
    L'agent voit cet env comme un env 1 joueur classique.

    Actions :
        0-8 -> cases de la grille de gauche à droite, haut en bas

    Récompenses (du point de vue de l'agent / joueur 0) :
        +1.0  victoire de l'agent
        -1.0  victoire du random
         0.0  nul ou coup intermédiaire

    État (27 valeurs, toujours du point de vue du joueur 0) :
        -  0.. 8 : cases occupées par l'agent (X)
        -  9..17 : cases occupées par le random (O)
        - 18..26 : cases vides
    """

    _WINNING_COMBOS = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6),
    ]

    def __init__(self, seed: int = None):
        self._board          = np.zeros(9, dtype=np.int8)
        self._done           = False
        self._winner: Optional[int] = None
        self._random         = RandomAgent(seed=seed)

    # ─────────────────────────────────────────────────────────────
    # Interface BaseEnv
    # ─────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        self._board  = np.zeros(9, dtype=np.int8)
        self._done   = False
        self._winner = None
        return self.get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        1. L'agent (X / valeur 1) joue `action`
        2. Si la partie continue, le random (O / valeur 2) joue
        3. Retourne (état, reward_agent, done)
        """
        assert not self._done, "Partie terminée, appelez reset()."
        assert action in self.available_actions(), f"Action illégale : {action}"

        # ── Coup de l'agent (joueur 0 = valeur 1) ────────────────
        self._board[action] = 1

        if self._check_winner(1):
            self._done   = True
            self._winner = 0
            return self.get_state(), 1.0, True

        if np.all(self._board != 0):
            self._done   = True
            self._winner = None
            return self.get_state(), 0.0, True

        # ── Coup du random (joueur 1 = valeur 2) ─────────────────
        random_action              = self._random.select_action(self)
        self._board[random_action] = 2

        if self._check_winner(2):
            self._done   = True
            self._winner = 1
            return self.get_state(), -1.0, True

        if np.all(self._board != 0):
            self._done   = True
            self._winner = None
            return self.get_state(), 0.0, True

        return self.get_state(), 0.0, False

    def available_actions(self) -> List[int]:
        if self._done:
            return []
        return [i for i in range(9) if self._board[i] == 0]

    def is_game_over(self) -> bool:
        return self._done

    def get_state(self) -> np.ndarray:
        """État toujours du point de vue du joueur 0 (l'agent)."""
        state = np.zeros(27, dtype=np.float32)
        for i in range(9):
            if self._board[i] == 1:        # case de l'agent
                state[i] = 1.0
            elif self._board[i] == 2:      # case du random
                state[9 + i] = 1.0
            else:                          # case vide
                state[18 + i] = 1.0
        return state

    def score(self) -> float:
        """
        Score final du point de vue de l'agent (joueur 0) :
            1.0  victoire
            0.5  nul
            0.0  défaite
        """
        if not self._done:
            raise ValueError("score() appelé avant la fin de la partie.")
        if self._winner == 0:
            return 1.0
        if self._winner is None:
            return 0.5
        return 0.0

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
            if self._winner == 0:
                print("\n→ Agent (X) gagne !")
            elif self._winner == 1:
                print("\n→ Random (O) gagne !")
            else:
                print("\n→ Match nul !")
        else:
            print("\n→ Tour de l'agent (X)")
        print()

    # ─────────────────────────────────────────────────────────────
    # Propriétés
    # ─────────────────────────────────────────────────────────────

    @property
    def state_size(self) -> int:
        return 27

    @property
    def action_size(self) -> int:
        return 9

    @property
    def current_player(self) -> int:
        return 0

    # ─────────────────────────────────────────────────────────────
    # Méthode privée
    # ─────────────────────────────────────────────────────────────

    def _check_winner(self, value: int) -> bool:
        for combo in self._WINNING_COMBOS:
            if all(self._board[i] == value for i in combo):
                return True
        return False