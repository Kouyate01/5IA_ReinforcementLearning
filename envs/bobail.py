import numpy as np
from typing import List, Tuple, Optional, Dict
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
        - 5 pions Joueu 0 (rangée du bas, ligne 4)
        - 5 pions Joueur 1 (rangée du haut, ligne 0)

    Disposition initiale :
        Ligne 0 : J1 J1 J1 J1 J1   (joueur 1)
        Ligne 1 : .  .  .  .  .
        Ligne 2 : .  .  B  .  .    (Bobail au centre)
        Ligne 3 : .  .  .  .  .
        Ligne 4 : J0 J0 J0 J0 J0   (joueur 0)

    Tour de jeu :
        Chaque tour se déroule en 2 phases :
        PHASE 1 → Déplacer le Bobail (1 case dans 8 directions)
        PHASE 2 → Déplacer un de ses propres pions (1 case dans 8 directions)

    Conditions de victoire :
        - Un joueur gagne s'il amène le Bobail sur sa rangée d'arrivée :
            Joueur 0 → amener le Bobail en ligne 0
            Joueur 1 → amener le Bobail en ligne 4
        - Un joueur gagne aussi si l'adversaire n'a plus de mouvements légaux.
        - Si un joueur ne peut pas déplacer le Bobail vers une case légale,
          il passe directement à la phase 2 (déplacement de pion).

    Mouvement :
        Les pions et le Bobail se déplacent d'une seule case dans les 8 directions.
        Un pion ne peut pas aller sur une case déjà occupée.
        Le Bobail ne peut pas aller sur une case occupée par un pion.

    ═══════════════════════════════════════════════════════════════
    ENCODING DE L'ÉTAT (taille = 77)
    ═══════════════════════════════════════════════════════════════
    Le plateau est représenté par 3 couches de 25 cases (5x5) aplaties :
        - Couche 0 (indices 0-24)  : 1.0 si case contient un pion Joueur 0
        - Couche 1 (indices 25-49) : 1.0 si case contient un pion Joueur 1
        - Couche 2 (indices 50-74) : 1.0 si case contient le Bobail
    + 2 valeurs scalaires :
        - Indice 75 : joueur courant (0.0 ou 1.0)
        - Indice 76 : phase courante (0.0=déplacer Bobail, 1.0=déplacer pion)
    → Taille totale : 77

    ═══════════════════════════════════════════════════════════════
    ENCODING DES ACTIONS
    ═══════════════════════════════════════════════════════════════
    Phase 1 (déplacer Bobail) : 8 directions depuis la position du Bobail
        Actions 0-7 : directions [N, NE, E, SE, S, SO, O, NO]

    Phase 2 (déplacer un pion) : 5 pions × 8 directions = 40 actions
        Action = pion_index * 8 + direction_index
        Actions 8-47

    → Taille totale de l'espace d'actions : 48
    (Beaucoup d'actions seront illégales selon l'état — utiliser available_actions())

    ═══════════════════════════════════════════════════════════════
    RÉCOMPENSES
    ═══════════════════════════════════════════════════════════════
        +1.0 → victoire (Bobail sur la rangée d'arrivée, ou adversaire bloqué)
        -1.0 → défaite
         0.0 → coup intermédiaire
    """

    # 8 directions : N, NE, E, SE, S, SO, O, NO
    _DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                   (1, 0),  (1, -1), (0, -1), (-1, -1)]

    # Lignes d'arrivée
    _WIN_ROW = {0: 0, 1: 4}  # joueur 0 veut amener le Bobail en ligne 0, joueur 1 en ligne 4

    def __init__(self):
        # Plateau : 0=vide, 1=pion joueur 0, 2=pion joueur 1, 3=Bobail
        self._board = np.zeros((5, 5), dtype=np.int8)
        self._bobail_pos: Tuple[int, int] = (2, 2)
        self._current_player: int = 0
        self._phase: int = 0  # 0=déplacer Bobail, 1=déplacer pion
        self._done: bool = False
        self._winner: Optional[int] = None
        self._reset_board()

    # -------------------------------------------------------------------------
    # Initialisation
    # -------------------------------------------------------------------------

    def _reset_board(self) -> None:
        """Met le plateau dans l'état initial."""
        self._board = np.zeros((5, 5), dtype=np.int8)
        # Pions joueur 0 en bas (ligne 4)
        for col in range(5):
            self._board[4][col] = 1
        # Pions joueur 1 en haut (ligne 0)
        for col in range(5):
            self._board[0][col] = 2
        # Bobail au centre
        self._board[2][2] = 3
        self._bobail_pos = (2, 2)

    # -------------------------------------------------------------------------
    # Méthodes abstraites obligatoires
    # -------------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        self._reset_board()
        self._current_player = 0
        self._phase = 0
        self._done = False
        self._winner = None
        return self.get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        assert not self._done, "Partie terminée, appelez reset()."
        assert action in self.available_actions(), (
            f"Action illégale : {action} | Phase={self._phase} | "
            f"Actions légales={self.available_actions()}"
        )

        reward = 0.0

        if self._phase == 0:
            # ── Phase 1 : déplacer le Bobail ──────────────────────────────
            direction = action  # 0-7
            br, bc = self._bobail_pos
            dr, dc = self._DIRECTIONS[direction]
            new_br, new_bc = br + dr, bc + dc

            # Déplacer le Bobail
            self._board[br][bc] = 0
            self._board[new_br][new_bc] = 3
            self._bobail_pos = (new_br, new_bc)

            # Vérifier victoire par position du Bobail
            win_row = self._WIN_ROW[self._current_player]
            if new_br == win_row:
                self._done = True
                self._winner = self._current_player
                reward = 1.0
                return self.get_state(), reward, self._done

            # Passer à la phase 2
            self._phase = 1

            # Si le joueur n'a aucun pion déplaçable en phase 2, on passe
            if len(self._get_phase2_actions()) == 0:
                self._end_turn()

        else:
            # ── Phase 2 : déplacer un pion ────────────────────────────────
            # action = 8 + pion_index * 8 + direction_index
            encoded = action - 8
            pion_idx = encoded // 8
            direction = encoded % 8 #(reste de la division par 8)

            pions = self._get_player_pions(self._current_player)
            pr, pc = pions[pion_idx]
            dr, dc = self._DIRECTIONS[direction]
            new_pr, new_pc = pr + dr, pc + dc

            # Déplacer le pion
            piece_val = self._board[pr][pc]
            self._board[pr][pc] = 0
            self._board[new_pr][new_pc] = piece_val

            # Fin du tour
            self._end_turn()

            # Vérifier si l'adversaire est bloqué
            if not self._done and len(self.available_actions()) == 0:
                self._done = True
                self._winner = self._current_player
                reward = 1.0

        return self.get_state(), reward, self._done

    def available_actions(self) -> List[int]:
        if self._done:
            return []

        if self._phase == 0:
            return self._get_phase1_actions()
        else:
            return self._get_phase2_actions()

    def is_game_over(self) -> bool:
        return self._done

    def get_state(self) -> np.ndarray:
        state = np.zeros(77, dtype=np.float32)
        for r in range(5):
            for c in range(5):
                idx = r * 5 + c
                cell = self._board[r][c]
                if cell == 1:      # pion joueur 0
                    state[idx] = 1.0
                elif cell == 2:    # pion joueur 1
                    state[25 + idx] = 1.0
                elif cell == 3:    # Bobail
                    state[50 + idx] = 1.0
        state[75] = float(self._current_player)
        state[76] = float(self._phase)
        return state

    def clone(self) -> "Bobail":
        return copy.deepcopy(self)

    def render(self) -> None:
        symbols = {0: ".", 1: "X", 2: "O", 3: "B"}
        player_names = {0: "Joueur 0 (X)", 1: "Joueur 1 (O)"}
        phase_names = {0: "déplacer Bobail", 1: "déplacer un pion"}

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
            print(f"\n→ {player_names[self._current_player]} | Phase: {phase_names[self._phase]}")
            print(f"  Bobail en {self._bobail_pos}")
        print()

    @property
    def state_size(self) -> int:
        return 77

    @property
    def action_size(self) -> int:
        return 48  # 8 (phase1) + 5*8 (phase2)

    @property
    def current_player(self) -> int:
        return self._current_player

    def score(self) -> float:
        """Score pour le joueur 0."""
        if self._winner == 0:
            return 1.0
        elif self._winner == 1:
            return 0.0
        return 0.5  # match nul (ne devrait pas arriver à Bobail)

    # -------------------------------------------------------------------------
    # Méthodes privées
    # -------------------------------------------------------------------------

    def _end_turn(self) -> None:
        """Change de joueur et remet la phase à 0."""
        self._current_player = 1 - self._current_player
        self._phase = 0

    def _get_player_pions(self, player: int) -> List[Tuple[int, int]]:
        """Retourne la liste des positions des pions d'un joueur, triée."""
        val = player + 1  # 1 pour joueur 0, 2 pour joueur 1
        positions = []
        for r in range(5):
            for c in range(5):
                if self._board[r][c] == val:
                    positions.append((r, c))
        return positions

    def _get_phase1_actions(self) -> List[int]:
        """Actions légales pour la phase 1 (déplacer le Bobail)."""
        actions = []
        br, bc = self._bobail_pos
        for d, (dr, dc) in enumerate(self._DIRECTIONS):
            nr, nc = br + dr, bc + dc
            if self._is_valid_bobail_move(nr, nc):
                actions.append(d)
        return actions

    def _get_phase2_actions(self) -> List[int]:
        """Actions légales pour la phase 2 (déplacer un pion du joueur courant)."""
        actions = []
        pions = self._get_player_pions(self._current_player)
        for pion_idx, (pr, pc) in enumerate(pions):
            for d, (dr, dc) in enumerate(self._DIRECTIONS):
                nr, nc = pr + dr, pc + dc
                if self._is_valid_pion_move(nr, nc):
                    actions.append(8 + pion_idx * 8 + d)
        return actions

    def _is_valid_bobail_move(self, r: int, c: int) -> bool:
        """Vérifie si le Bobail peut aller en (r, c)."""
        if not (0 <= r < 5 and 0 <= c < 5):
            return False
        return self._board[r][c] == 0  # doit être vide

    def _is_valid_pion_move(self, r: int, c: int) -> bool:
        """Vérifie si un pion peut aller en (r, c)."""
        if not (0 <= r < 5 and 0 <= c < 5):
            return False
        return self._board[r][c] == 0  # doit être vide (pas de capture)

    def get_bobail_position(self) -> Tuple[int, int]:
        """Retourne la position actuelle du Bobail."""
        return self._bobail_pos

    def get_board(self) -> np.ndarray:
        """Retourne une copie du plateau (pour la GUI)."""
        return self._board.copy()

    def get_phase(self) -> int:
        """Retourne la phase courante (0 ou 1)."""
        return self._phase