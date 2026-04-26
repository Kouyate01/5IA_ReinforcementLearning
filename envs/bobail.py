import numpy as np
from typing import List, Tuple, Optional
import copy
from envs.base_env import BaseEnv


class Bobail(BaseEnv):
    """
    Environnement Bobail.

    ═══════════════════════════════════════════════════════════════
    RÈGLES DU JEU
    ═══════════════════════════════════════════════════════════════
    Plateau : grille 5x5
    Pièces :
        - 1 Bobail (pièce neutre, partagée)
        - 5 pions Joueur 0 (rangée du bas, ligne 4)
        - 5 pions Joueur 1 (rangée du haut, ligne 0)

    Disposition initiale :
        Ligne 0 : J1 J1 J1 J1 J1   (joueur 1)
        Ligne 1 : .  .  .  .  .
        Ligne 2 : .  .  B  .  .    (Bobail au centre)
        Ligne 3 : .  .  .  .  .
        Ligne 4 : J0 J0 J0 J0 J0   (joueur 0)

    Tour de jeu :
        PREMIER TOUR (J0 uniquement) :
            → J0 déplace seulement un pion (phase 1 directement, pas de Bobail)

        Tours suivants (chaque tour en 2 phases) :
        PHASE 0 → Déplacer le Bobail (1 case dans 8 directions, case vide uniquement)
                  Si le Bobail ne peut PAS être déplacé → défaite immédiate du joueur courant
        PHASE 1 → Déplacer un de ses propres pions :
                  Le pion glisse jusqu'à la case la plus éloignée possible dans
                  une direction (comme une tour aux échecs), sans sauter par-dessus
                  d'autres pions ni par-dessus le Bobail.

    Conditions de victoire :
        - Amener le Bobail sur sa propre ligne de base :
            Joueur 0 → ligne 4  |  Joueur 1 → ligne 0
        - Enfermer le Bobail (aucun mouvement possible pour le Bobail au début
          du tour adverse) → le joueur qui vient de jouer gagne.
        - Si le Bobail ne peut pas être déplacé en début de tour → défaite
          immédiate du joueur courant.

    ═══════════════════════════════════════════════════════════════
    ENCODING DE L'ÉTAT (taille = 78)
    ═══════════════════════════════════════════════════════════════
        - Couche 0 (indices  0-24) : 1.0 si case contient un pion Joueur 0
        - Couche 1 (indices 25-49) : 1.0 si case contient un pion Joueur 1
        - Couche 2 (indices 50-74) : 1.0 si case contient le Bobail
        - Indice 75 : joueur courant (0.0 ou 1.0)
        - Indice 76 : phase courante (0.0 = déplacer Bobail, 1.0 = déplacer pion)
        - Indice 77 : premier tour (1.0 si c'est le tout premier tour, sinon 0.0)
    → Taille totale : 78

    ═══════════════════════════════════════════════════════════════
    ENCODING DES ACTIONS
    ═══════════════════════════════════════════════════════════════
    Phase 0 — déplacer le Bobail (8 actions) :
        action = direction    avec direction ∈ [0, 7]
        Directions : 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SO, 6=O, 7=NO

    Phase 1 — déplacer un pion (200 actions) :
        action = 8 + (row * 5 + col) * 8 + direction
        → 25 cases × 8 directions = 200 actions possibles

        Décodage :
            encoded   = action - 8
            case_idx  = encoded // 8     → index de la case de départ (row*5+col)
            direction = encoded  % 8     → direction du glissement
            row       = case_idx // 5
            col       = case_idx  % 5

    → Taille totale de l'espace d'actions : 208

    ═══════════════════════════════════════════════════════════════
    RÉCOMPENSES
    ═══════════════════════════════════════════════════════════════
        +1.0 → victoire
        -1.0 → défaite
         0.0 → coup intermédiaire
    """

    _DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                   (1, 0),  (1, -1), (0, -1), (-1, -1)]

    _WIN_ROW = {0: 4, 1: 0}

    def __init__(self):
        self._board = np.zeros((5, 5), dtype=np.int8)
        self._bobail_pos: Tuple[int, int] = (2, 2)
        self._current_player: int = 0
        self._phase: int = 1          # J0 commence directement en phase 1
        self._first_turn: bool = True  # Marqueur premier tour
        self._done: bool = False
        self._winner: Optional[int] = None
        self._reset_board()

    def _reset_board(self) -> None:
        self._board = np.zeros((5, 5), dtype=np.int8)
        for col in range(5):
            self._board[4][col] = 1   # J0 ligne 4
        for col in range(5):
            self._board[0][col] = 2   # J1 ligne 0
        self._board[2][2] = 3         # Bobail centre
        self._bobail_pos = (2, 2)

    def reset(self) -> np.ndarray:
        self._reset_board()
        self._current_player = 0
        self._phase      = 1     # Premier tour : J0 commence en phase 1
        self._first_turn = True
        self._done       = False
        self._winner     = None
        return self.get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        assert not self._done, "Partie terminée, appelez reset()."
        assert action in self.available_actions(), (
            f"Action illégale : {action} | Phase={self._phase} | "
            f"Actions légales={self.available_actions()}"
        )

        reward = 0.0

        # ── Phase 0 : déplacer le Bobail ─────────────────────────────────────
        if self._phase == 0:
            direction = action
            br, bc = self._bobail_pos
            dr, dc = self._DIRECTIONS[direction]
            new_br, new_bc = br + dr, bc + dc

            self._board[br][bc] = 0
            self._board[new_br][new_bc] = 3
            self._bobail_pos = (new_br, new_bc)

            # Victoire : Bobail sur la ligne de base du joueur courant
            if new_br == self._WIN_ROW[self._current_player]:
                self._done   = True
                self._winner = self._current_player
                return self.get_state(), 1.0, True

            # Bobail sur la ligne de base adverse → l'adversaire gagne
            opponent = 1 - self._current_player
            if new_br == self._WIN_ROW[opponent]:
                self._done   = True
                self._winner = opponent
                return self.get_state(), -1.0, True

            self._phase = 1

            if len(self._get_phase1_actions()) == 0:
                self._end_turn()
                self._check_bobail_blocked_defeat()

        # ── Phase 1 : déplacer un pion ────────────────────────────────────────
        else:
            encoded   = action - 8
            case_idx  = encoded // 8
            direction = encoded  % 8
            from_row  = case_idx // 5
            from_col  = case_idx  % 5

            to_row, to_col = self._slide_destination(from_row, from_col, direction)

            piece_val = self._board[from_row][from_col]
            self._board[from_row][from_col] = 0
            self._board[to_row][to_col]     = piece_val

            # Fin du premier tour : J1 joue normalement dès maintenant
            if self._first_turn:
                self._first_turn = False

            self._end_turn()
            self._check_bobail_blocked_defeat()

            # Bobail enfermé → le joueur qui vient de jouer gagne
            if not self._done and len(self._get_phase0_actions()) == 0:
                self._done   = True
                self._winner = 1 - self._current_player

        return self.get_state(), reward, self._done

    def available_actions(self) -> List[int]:
        if self._done:
            return []
        if self._phase == 0:
            return self._get_phase0_actions()
        else:
            return self._get_phase1_actions()

    def is_game_over(self) -> bool:
        return self._done

    def get_state(self) -> np.ndarray:
        state = np.zeros(78, dtype=np.float32)
        for r in range(5):
            for c in range(5):
                idx  = r * 5 + c
                cell = self._board[r][c]
                if cell == 1:
                    state[idx] = 1.0
                elif cell == 2:
                    state[25 + idx] = 1.0
                elif cell == 3:
                    state[50 + idx] = 1.0
        state[75] = float(self._current_player)
        state[76] = float(self._phase)
        state[77] = 1.0 if self._first_turn else 0.0
        return state

    def clone(self) -> "Bobail":
        return copy.deepcopy(self)

    def render(self) -> None:
        symbols      = {0: ".", 1: "X", 2: "O", 3: "B"}
        player_names = {0: "Joueur 0 (X)", 1: "Joueur 1 (O)"}
        phase_names  = {0: "déplacer Bobail", 1: "déplacer un pion"}

        print("\n  0 1 2 3 4")
        print(" +----------")
        for r in range(5):
            row_str = f"{r}|"
            for c in range(5):
                row_str += symbols[self._board[r][c]] + " "
            print(row_str)

        if self._done:
            if self._winner is not None:
                print(f"\n→ {player_names[self._winner]} gagne !")
            else:
                print("\n→ Match nul !")
        else:
            ft = " [PREMIER TOUR → pas de Bobail]" if self._first_turn else ""
            print(f"\n→ {player_names[self._current_player]} "
                  f"| Phase : {phase_names[self._phase]}{ft}")
            print(f"  Bobail en {self._bobail_pos}")
        print()

    # ─────────────────────────────────────────────────────────────────────────
    # Propriétés
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def state_size(self) -> int:
        return 78

    @property
    def action_size(self) -> int:
        return 208  # 8 (phase0) + 25*8 (phase1)

    @property
    def current_player(self) -> int:
        return self._current_player

    def score(self) -> float:
        if self._winner == 0:
            return 1.0
        elif self._winner == 1:
            return 0.0
        return 0.5

    # ─────────────────────────────────────────────────────────────────────────
    # Méthodes internes
    # ─────────────────────────────────────────────────────────────────────────

    def _end_turn(self) -> None:
        self._current_player = 1 - self._current_player
        self._phase = 0

    def _check_bobail_blocked_defeat(self) -> None:
        """
        Si le joueur courant ne peut pas déplacer le Bobail en phase 0,
        il perd immédiatement.
        """
        if not self._done and self._phase == 0:
            if len(self._get_phase0_actions()) == 0:
                self._done   = True
                self._winner = 1 - self._current_player

    def _slide_destination(self, from_row: int, from_col: int,
                           direction: int) -> Tuple[int, int]:
        """
        Glissement : le pion avance dans la direction jusqu'à la dernière
        case vide avant un obstacle ou le bord.
        """
        dr, dc   = self._DIRECTIONS[direction]
        cur_r, cur_c = from_row, from_col
        while True:
            next_r = cur_r + dr
            next_c = cur_c + dc
            if not (0 <= next_r < 5 and 0 <= next_c < 5):
                break
            if self._board[next_r][next_c] != 0:
                break
            cur_r, cur_c = next_r, next_c
        return cur_r, cur_c

    def _get_phase0_actions(self) -> List[int]:
        """Actions légales pour déplacer le Bobail (phase 0)."""
        actions = []
        br, bc  = self._bobail_pos
        for d, (dr, dc) in enumerate(self._DIRECTIONS):
            nr, nc = br + dr, bc + dc
            if 0 <= nr < 5 and 0 <= nc < 5 and self._board[nr][nc] == 0:
                actions.append(d)
        return actions

    def _get_phase1_actions(self) -> List[int]:
        return self._get_phase2_actions()

    def _get_phase2_actions(self) -> List[int]:
        """
        Actions légales pour déplacer un pion (phase 1).
        action = 8 + (row*5+col)*8 + direction
        """
        actions = []
        for (pr, pc) in self._get_player_pions(self._current_player):
            case_idx = pr * 5 + pc
            for d, (dr, dc) in enumerate(self._DIRECTIONS):
                nr, nc = pr + dr, pc + dc
                if 0 <= nr < 5 and 0 <= nc < 5 and self._board[nr][nc] == 0:
                    actions.append(8 + case_idx * 8 + d)
        return actions

    def _get_player_pions(self, player: int) -> List[Tuple[int, int]]:
        val = player + 1
        return [
            (r, c)
            for r in range(5)
            for c in range(5)
            if self._board[r][c] == val
        ]

    def get_bobail_position(self) -> Tuple[int, int]:
        return self._bobail_pos

    def get_board(self) -> np.ndarray:
        return self._board.copy()

    def get_phase(self) -> int:
        return self._phase

    @property
    def is_first_turn(self) -> bool:
        return self._first_turn
