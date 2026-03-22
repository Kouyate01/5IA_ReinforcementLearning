"""
╔══════════════════════════════════════════════════════════════════╗
║         TicTacToe vs Random — Environnement RL                  ║
║         Projet Deep Reinforcement Learning                       ║
╚══════════════════════════════════════════════════════════════════╝

Référence : Sutton & Barto, "Reinforcement Learning: An Introduction"

Ce module implémente :
  ① L'environnement TicTacToe (interface gym-like)
  ② Un agent Random (baseline)
  ③ Un benchmark de simulation (parties/seconde)
  ④ La documentation complète des encodings état/action
"""

import numpy as np
import random
import time
from typing import Optional, List, Tuple, Dict


# ══════════════════════════════════════════════════════════════════
#  SECTION 1 — ENCODINGS (à lire avant tout)
# ══════════════════════════════════════════════════════════════════

"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DESCRIPTION DE L'ÉTAT — State Encoding
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Le plateau 3×3 est représenté comme un vecteur PLAT de longueur 9.

Indexation des cases :
    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8

Encodage par case :
    0  → case vide
    1  → joueur 1 (agent RL, symbole X)
   -1  → joueur 2 (adversaire random, symbole O)

Exemple de plateau :
    X | . | O          indices : [0, 1, 2, 3, 4, 5, 6, 7, 8]
    . | X | .    →     état   : [1, 0,-1, 0, 1, 0, 0, 0,-1]
    . | . | O

Propriétés du vecteur d'état :
    - Taille                    : 9
    - Type                      : np.float32
    - Valeurs par dimension     : {-1.0, 0.0, 1.0}
    - Nombre d'états théoriques : 3^9 = 19 683 (dont ~5 478 légaux)
    - Symétries exploitables    : 8 (rotations + réflexions)

Pour le TabularQLearning, l'état peut être converti en clé :
    tuple(state.astype(int))  → ex: (1, 0, -1, 0, 1, 0, 0, 0, -1)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DESCRIPTION D'UNE ACTION — Action Encoding
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Une action = choisir une case libre parmi les 9.

Encodage entier (pour Q-Learning, Q-table) :
    action ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8}
    correspondance : case (row, col) → action = row * 3 + col

Encodage one-hot (pour réseaux de neurones) :
    vecteur de longueur 9, ex: action 4 (centre) :
    [0, 0, 0, 0, 1, 0, 0, 0, 0]

Propriétés de l'espace d'actions :
    - Type                  : discret
    - Taille totale         : 9
    - Actions légales       : cases où state[i] == 0 (variable selon plateau)
    - Actions illégales     : cases déjà occupées (à masquer dans le réseau)

Masque d'actions légales (action_mask) :
    np.array([1 si state[i]==0 else 0 for i in range(9)], dtype=np.float32)
    → multiplier la sortie du réseau par ce masque avant argmax/softmax
"""

ACTION_SPACE_SIZE = 9   # taille de l'espace d'actions
STATE_SPACE_SIZE  = 9   # taille du vecteur d'état


# ══════════════════════════════════════════════════════════════════
#  SECTION 2 — ENVIRONNEMENT
# ══════════════════════════════════════════════════════════════════

class TicTacToeEnv:
    """
    Environnement TicTacToe (interface gym-like) pour le RL.

    L'agent joue toujours comme joueur +1 (X).
    L'adversaire est un agent random qui joue comme joueur -1 (O).

    Cycle de vie :
        state = env.reset()
        while not done:
            action = agent.select_action(state, env.get_valid_actions())
            next_state, reward, done, info = env.step(action)
            state = next_state

    Récompenses (sparse rewards — cf. Sutton & Barto, Ch. 3) :
        +1.0  → l'agent gagne
        -1.0  → l'adversaire gagne
         0.0  → match nul ou coup intermédiaire
    """

    # Les 8 lignes gagnantes (indices dans le vecteur plat)
    WINNING_LINES: List[List[int]] = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],   # lignes horizontales
        [0, 3, 6], [1, 4, 7], [2, 5, 8],   # colonnes verticales
        [0, 4, 8], [2, 4, 6],              # diagonales
    ]

    def __init__(self, agent_starts: bool = True):
        """
        Paramètres
        ----------
        agent_starts : bool
            True  → l'agent joue en premier (X commence)
            False → l'adversaire random commence
        """
        self.agent_starts = agent_starts
        self.board         = np.zeros(9, dtype=np.int8)
        self.done          = False
        self.winner: Optional[int] = None  # 1=agent, -1=random, 0=nul

    # ── Interface principale ────────────────────────────────────

    def reset(self) -> np.ndarray:
        """
        Réinitialise l'environnement.

        Retourne
        --------
        state : np.ndarray shape (9,), dtype float32
            Vecteur d'état initial vu par l'agent.
        """
        self.board  = np.zeros(9, dtype=np.int8)
        self.done   = False
        self.winner = None

        # Si l'adversaire commence, il joue immédiatement
        if not self.agent_starts:
            self._play_random(player=-1)

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Exécute l'action de l'agent, puis le coup adverse (random).

        Paramètres
        ----------
        action : int ∈ [0..8]
            Case choisie par l'agent (doit être libre).

        Retourne
        --------
        next_state : np.ndarray (9,)  — état après les deux coups
        reward     : float            — +1 / -1 / 0
        done       : bool             — True si la partie est terminée
        info       : dict             — infos de débogage
        """
        if self.done:
            raise RuntimeError("La partie est terminée. Appelez reset().")
        if not self._is_valid_action(action):
            raise ValueError(f"Action illégale : {action}. "
                             f"Cases libres : {self.get_valid_actions()}")

        # ── Coup de l'agent (joueur +1) ──
        self.board[action] = 1

        if self._check_winner(1):
            self.done, self.winner = True, 1
            return self._get_state(), 1.0, True, {"winner": "agent", "result": "win"}

        if self._board_full():
            self.done, self.winner = True, 0
            return self._get_state(), 0.0, True, {"winner": None, "result": "draw"}

        # ── Coup de l'adversaire random (joueur -1) ──
        self._play_random(player=-1)

        if self._check_winner(-1):
            self.done, self.winner = True, -1
            return self._get_state(), -1.0, True, {"winner": "random", "result": "loss"}

        if self._board_full():
            self.done, self.winner = True, 0
            return self._get_state(), 0.0, True, {"winner": None, "result": "draw"}

        return self._get_state(), 0.0, False, {"winner": None, "result": "ongoing"}

    # ── Helpers état / action ───────────────────────────────────

    def _get_state(self) -> np.ndarray:
        """Vecteur d'état courant, copie en float32."""
        return self.board.copy().astype(np.float32)

    def get_valid_actions(self) -> List[int]:
        """Retourne la liste des indices des cases libres."""
        return [i for i in range(9) if self.board[i] == 0]

    def get_action_mask(self) -> np.ndarray:
        """
        Masque binaire des actions légales.
        mask[i] = 1.0 si la case i est libre, 0.0 sinon.
        À multiplier avec la sortie du réseau avant argmax.
        """
        return (self.board == 0).astype(np.float32)

    def action_to_onehot(self, action: int) -> np.ndarray:
        """Encode l'action en vecteur one-hot de longueur 9."""
        v = np.zeros(9, dtype=np.float32)
        v[action] = 1.0
        return v

    def state_description(self) -> Dict:
        """Résumé lisible de l'état courant (utile pour le débogage)."""
        return {
            "board_flat"    : self._get_state().tolist(),
            "board_2d"      : self.board.reshape(3, 3).tolist(),
            "valid_actions" : self.get_valid_actions(),
            "action_mask"   : self.get_action_mask().tolist(),
            "done"          : self.done,
            "winner"        : self.winner,
        }

    # ── Méthodes internes ───────────────────────────────────────

    def _is_valid_action(self, a: int) -> bool:
        return 0 <= a < 9 and self.board[a] == 0

    def _board_full(self) -> bool:
        return np.all(self.board != 0)

    def _check_winner(self, player: int) -> bool:
        for line in self.WINNING_LINES:
            if self.board[line[0]] == player \
            and self.board[line[1]] == player \
            and self.board[line[2]] == player:
                return True
        return False

    def _play_random(self, player: int) -> None:
        valids = self.get_valid_actions()
        if valids:
            self.board[random.choice(valids)] = player

    # ── Rendu console ───────────────────────────────────────────

    def render(self) -> str:
        """Retourne une représentation ASCII du plateau."""
        sym = {1: "X", -1: "O", 0: "."}
        rows = []
        for r in range(3):
            row = " │ ".join(sym[int(self.board[r * 3 + c])] for c in range(3))
            rows.append(row)
        sep = "\n──┼───┼──\n"
        return sep.join(rows)


# ══════════════════════════════════════════════════════════════════
#  SECTION 3 — AGENTS
# ══════════════════════════════════════════════════════════════════

class RandomAgent:
    """
    Agent baseline : choisit uniformément parmi les actions légales.
    Référence : politique aléatoire π(a|s) = 1/|A(s)|
    (Sutton & Barto, §1.1 — référence de départ pour tout benchmark)
    """

    def select_action(self,
                      state: np.ndarray,
                      valid_actions: List[int]) -> int:
        return random.choice(valid_actions)


class HumanAgent:
    """Agent interactif (saisie console)."""

    def select_action(self,
                      state: np.ndarray,
                      valid_actions: List[int]) -> int:
        print(f"Cases libres : {valid_actions}")
        while True:
            try:
                action = int(input("Votre coup (0-8) : "))
                if action in valid_actions:
                    return action
                print(f"  ✗ Case occupée. Choisissez parmi {valid_actions}.")
            except ValueError:
                print("  ✗ Entrez un entier entre 0 et 8.")


# ══════════════════════════════════════════════════════════════════
#  SECTION 4 — SIMULATION & BENCHMARK
# ══════════════════════════════════════════════════════════════════

def run_simulation(n_games: int = 100_000,
                   verbose: bool = True) -> Dict:
    """
    Simule n_games parties (agent Random vs adversaire Random).

    But : mesurer les performances de l'environnement (parties/seconde)
    et vérifier la distribution des résultats sous politique aléatoire.

    Paramètres
    ----------
    n_games : int    — nombre de parties à simuler
    verbose : bool   — affiche les résultats dans le terminal

    Retourne
    --------
    dict avec wins, draws, losses, games_per_second, ...
    """
    env   = TicTacToeEnv(agent_starts=True)
    agent = RandomAgent()

    wins = draws = losses = total_steps = 0

    t_start = time.perf_counter()

    for _ in range(n_games):
        state = env.reset()
        done  = False
        steps = 0
        while not done:
            action = agent.select_action(state, env.get_valid_actions())
            state, reward, done, _ = env.step(action)
            steps += 1
        total_steps += steps

        if   reward ==  1.0: wins   += 1
        elif reward == -1.0: losses += 1
        else:                draws  += 1

    elapsed = time.perf_counter() - t_start
    gps     = n_games / elapsed

    results = {
        "n_games"          : n_games,
        "wins"             : wins,
        "draws"            : draws,
        "losses"           : losses,
        "win_rate"         : wins   / n_games,
        "draw_rate"        : draws  / n_games,
        "loss_rate"        : losses / n_games,
        "avg_steps"        : total_steps / n_games,
        "elapsed_sec"      : round(elapsed, 4),
        "games_per_second" : round(gps, 1),
    }

    if verbose:
        print("╔" + "═" * 45 + "╗")
        print(f"║  BENCHMARK — {n_games:>10,} parties simulées       ║")
        print("╠" + "═" * 45 + "╣")
        print(f"║  Victoires (agent)  : {wins:>8,}  ({wins/n_games*100:5.1f} %)  ║")
        print(f"║  Défaites  (random) : {losses:>8,}  ({losses/n_games*100:5.1f} %)  ║")
        print(f"║  Matchs nuls        : {draws:>8,}  ({draws/n_games*100:5.1f} %)  ║")
        print("╠" + "═" * 45 + "╣")
        print(f"║  Durée totale  : {elapsed:>8.3f} s                  ║")
        print(f"║  Parties/sec   : {gps:>10,.0f}                  ║")
        print(f"║  Coups moyens  : {total_steps/n_games:>8.2f} coups/partie      ║")
        print("╚" + "═" * 45 + "╝")

    return results


def play_vs_human() -> None:
    """Lance une partie joueur humain contre adversaire random (console)."""
    env   = TicTacToeEnv(agent_starts=True)
    human = HumanAgent()

    print("\n══════════════════════════════")
    print("  TicTacToe — Vous jouez X")
    print("  (l'adversaire O est random)")
    print("══════════════════════════════")
    print("\nIndexation des cases :")
    print("  0 │ 1 │ 2")
    print("  ──┼───┼──")
    print("  3 │ 4 │ 5")
    print("  ──┼───┼──")
    print("  6 │ 7 │ 8\n")

    state = env.reset()
    done  = False

    while not done:
        print(env.render())
        print()
        action = human.select_action(state, env.get_valid_actions())
        state, reward, done, info = env.step(action)

    print(env.render())
    print()
    if   reward ==  1.0: print("🎉 Vous avez gagné !")
    elif reward == -1.0: print("😞 L'adversaire a gagné.")
    else:                print("🤝 Match nul.")


# ══════════════════════════════════════════════════════════════════
#  SECTION 5 — DÉMONSTRATIONS D'ENCODING
# ══════════════════════════════════════════════════════════════════

def demo_encodings() -> None:
    """Affiche des exemples concrets d'encodage état/action."""
    print("\n" + "═" * 55)
    print("  ENCODINGS — Exemples concrets")
    print("═" * 55)

    env = TicTacToeEnv()

    # Situation initiale
    state = env.reset()
    print("\n▶ ÉTAT INITIAL")
    print(env.render())
    print(f"  Vecteur d'état  : {state}")
    print(f"  Actions légales : {env.get_valid_actions()}")
    print(f"  Masque d'actions: {env.get_action_mask()}")

    # Après quelques coups
    env.board = np.array([1, 0, -1, 0, 1, 0, 0, 0, -1], dtype=np.int8)
    state = env._get_state()
    print("\n▶ ÉTAT INTERMÉDIAIRE")
    print(env.render())
    print(f"  Vecteur d'état  : {state}")
    print(f"  Actions légales : {env.get_valid_actions()}")
    print(f"  Masque d'actions: {env.get_action_mask()}")

    # Exemples d'encodage d'actions
    print("\n▶ ACTIONS — Exemples d'encodage one-hot")
    for action, name in [(0, "coin haut-gauche"), (4, "centre"), (8, "coin bas-droit")]:
        oh = env.action_to_onehot(action)
        print(f"  action {action} ({name}): {oh}")

    # Clé pour Q-table
    print("\n▶ CLÉ POUR Q-TABLE (TabularQLearning)")
    key = tuple(state.astype(int))
    print(f"  tuple(state) = {key}")
    print("  → Nombre d'états légaux ≈ 5 478")
    print("  → Q-table de taille : 5 478 × 9")
    print()


# ══════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("bench", "all"):
        run_simulation(n_games=100_000)

    if mode in ("enc", "all"):
        demo_encodings()

    if mode == "human":
        play_vs_human()
