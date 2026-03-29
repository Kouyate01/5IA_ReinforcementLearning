import numpy as np
from typing import List, Tuple, Optional
import copy
from envs.base_env import BaseEnv


class TicTacToe(BaseEnv):
    """
    Environnement TicTacToe (Morpion) 3x3.
    
    Deux joueurs s'affrontent :
        - Joueur 0 joue les X
        - Joueur 1 joue les O (peut être un agent Random)
    
    Actions:
        0-8 → cases de la grille, numérotées de gauche à droite, haut en bas :
            0 | 1 | 2
            ---------
            3 | 4 | 5
            ---------
            6 | 7 | 8
    
    Récompenses (du point de vue du joueur qui vient de jouer) :
        +1.0  → victoire
        -1.0  → défaite
         0.5  → match nul
         0.0  → coup intermédiaire
    
    State encoding (taille 27) :
        - Indices 0-8  : 1.0 si case occupée par joueur 0, 0 sinon
        - Indices 9-17 : 1.0 si case occupée par joueur 1, 0 sinon
        - Indices 18-26: 1.0 si case vide, 0 sinon
        → Taille totale : 27
        
    Note: On n'encode PAS le joueur courant dans le vecteur d'état,
    car le réseau de neurones jouera toujours du point de vue du joueur courant
    (on peut retourner le vecteur si nécessaire).
    """

    # Combinaisons gagnantes (indices de cases)
    _WINNING_COMBOS = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # lignes
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # colonnes
        (0, 4, 8), (2, 4, 6),              # diagonales
    ]

    def __init__(self):
        self._board = np.zeros(9, dtype=np.int8)  # 0=vide, 1=joueur0, 2=joueur1
        self._current_player = 0
        self._done = False
        self._winner: Optional[int] = None  # 0, 1, ou None (nul)

    # -------------------------------------------------------------------------
    # Implémentation des méthodes abstraites
    # -------------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        self._board = np.zeros(9, dtype=np.int8)
        self._current_player = 0
        self._done = False
        self._winner = None
        return self.get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        assert not self._done, "La partie est terminée, appelez reset()."
        assert action in self.available_actions(), f"Action illégale : {action}"

        # Placer le pion
        self._board[action] = self._current_player + 1  # 1 ou 2

        # Vérifier victoire
        reward = 0.0
        if self._check_winner(self._current_player):
            self._done = True
            self._winner = self._current_player
            reward = 1.0
        elif len(self.available_actions()) == 0:
            # Match nul (on vérifie APRÈS avoir placé, donc on re-check)
            empty = np.sum(self._board == 0)
            if empty == 0:
                self._done = True
                self._winner = None
                reward = 0.5
        
        # Changer de joueur si la partie continue
        if not self._done:
            self._current_player = 1 - self._current_player

        return self.get_state(), reward, self._done

    def available_actions(self) -> List[int]:
        if self._done:
            return []
        return [i for i in range(9) if self._board[i] == 0]

    def is_game_over(self) -> bool:
        return self._done

    def get_state(self) -> np.ndarray:
        state = np.zeros(27, dtype=np.float32)
        for i in range(9):
            if self._board[i] == 1:    # joueur 0
                state[i] = 1.0
            elif self._board[i] == 2:  # joueur 1
                state[9 + i] = 1.0
            else:                      # vide
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
        """Score pour le joueur 0 : 1=victoire, 0.5=nul, 0=défaite."""
        if self._winner == 0:
            return 1.0
        elif self._winner is None and self._done:
            return 0.5
        return 0.0

    # -------------------------------------------------------------------------
    # Méthodes privées
    # -------------------------------------------------------------------------

    def _check_winner(self, player: int) -> bool:
        """Vérifie si le joueur donné a gagné."""
        val = player + 1  # 1 pour joueur 0, 2 pour joueur 1
        for combo in self._WINNING_COMBOS:
            if all(self._board[i] == val for i in combo):
                return True
        return False