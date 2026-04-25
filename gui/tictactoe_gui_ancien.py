"""
gui/tictactoe_gui.py
====================
Interface graphique tkinter pour l'environnement TicTacToe.

Modes disponibles :
  • Human vs Random  — l'humain joue X, l'IA joue O au hasard
  • Human vs Human   — 2 joueurs sur le même clavier, alternance X/O
  • Random vs Random — auto-step, les deux joueurs jouent au hasard
  • Simulation       — benchmark N parties R‑vs‑R avec stats live

Design : fond sombre #0e0f14, accents or #e8c547, X rouge #e05c5c, O bleu #5ca8e0
"""

import sys
import os
import time
import random
import threading
import numpy as np
import tkinter as tk
from tkinter import ttk, font as tkfont

# ── import de l'environnement ────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.tictactoe import TicTacToe 

from agents.tabular_q_learning import TabularQLearning
from agents.dqn import DeepQLearning
from agents.ddqn import DoubleDeepQLearning
from agents.ddqner import DoubleDeepQLearningWithExperienceReplay
from agents.ddqnper import DoubleDeepQLearningWithPrioritizedExperienceReplay

from train import make_agent, load_model

# ── Palette de couleurs ───────────────────────────────────────────────────────
BG        = "#0e0f14"
BG2       = "#16171f"
BG3       = "#1e1f2b"
GOLD      = "#e8c547"
GOLD_DIM  = "#a0893a"
X_COL     = "#e05c5c"
O_COL     = "#5ca8e0"
TXT       = "#d4d4d8"
TXT_DIM   = "#6b7280"
WIN_LINE  = "#e8c547"

CELL_SIZE = 120
PADDING   = 12
BOARD_CANVAS_SIZE = CELL_SIZE * 3 + PADDING * 2

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_winning_combo(board: np.ndarray, player: int):
    """Retourne les indices de la combinaison gagnante ou None."""
    val = player + 1
    for combo in TicTacToe._WINNING_COMBOS:
        if all(board[i] == val for i in combo):
            return combo
    return None


def cell_center(idx: int):
    """Retourne (x, y) du centre de la case idx sur le canvas."""
    row, col = divmod(idx, 3)
    x = PADDING + col * CELL_SIZE + CELL_SIZE // 2
    y = PADDING + row * CELL_SIZE + CELL_SIZE // 2
    return x, y


# ─────────────────────────────────────────────────────────────────────────────
# Widget : plateau de jeu (canvas)
# ─────────────────────────────────────────────────────────────────────────────

class BoardCanvas(tk.Canvas):
    def __init__(self, parent, click_callback=None, **kwargs):
        size = BOARD_CANVAS_SIZE
        super().__init__(
            parent,
            width=size, height=size,
            bg=BG2, highlightthickness=0,
            **kwargs
        )
        self._click_cb = click_callback
        self.bind("<Button-1>", self._on_click)
        self._draw_grid()

    def _draw_grid(self):
        self.delete("grid")
        s = BOARD_CANVAS_SIZE
        p = PADDING
        # lignes verticales
        for col in range(1, 3):
            x = p + col * CELL_SIZE
            self.create_line(x, p, x, s - p, fill=BG3, width=3, tags="grid")
        # lignes horizontales
        for row in range(1, 3):
            y = p + row * CELL_SIZE
            self.create_line(p, y, s - p, y, fill=BG3, width=3, tags="grid")

    def _on_click(self, event):
        if self._click_cb is None:
            return
        col = (event.x - PADDING) // CELL_SIZE
        row = (event.y - PADDING) // CELL_SIZE
        if 0 <= row < 3 and 0 <= col < 3:
            idx = row * 3 + col
            self._click_cb(idx)

    def render_board(self, board: np.ndarray, winning_combo=None):
        self.delete("piece", "winline")
        for idx in range(9):
            val = board[idx]
            if val == 0:
                continue
            cx, cy = cell_center(idx)
            color = X_COL if val == 1 else O_COL
            symbol = "X" if val == 1 else "O"
            r = CELL_SIZE // 2 - 20
            if symbol == "X":
                self.create_line(cx - r, cy - r, cx + r, cy + r,
                                 fill=color, width=6, capstyle=tk.ROUND, tags="piece")
                self.create_line(cx + r, cy - r, cx - r, cy + r,
                                 fill=color, width=6, capstyle=tk.ROUND, tags="piece")
            else:
                self.create_oval(cx - r, cy - r, cx + r, cy + r,
                                 outline=color, width=6, tags="piece")

        # Ligne gagnante
        if winning_combo:
            ax, ay = cell_center(winning_combo[0])
            bx, by = cell_center(winning_combo[2])
            self.create_line(ax, ay, bx, by,
                             fill=WIN_LINE, width=5, capstyle=tk.ROUND, tags="winline")

    def clear(self):
        self.delete("piece", "winline")


# ─────────────────────────────────────────────────────────────────────────────
# Widget : barre de statistiques de session
# ─────────────────────────────────────────────────────────────────────────────

class SessionStatsBar(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG, **kwargs)
        self.wins = 0
        self.losses = 0
        self.draws = 0

        lbl_cfg = dict(bg=BG, fg=TXT, font=("Segoe UI", 10))
        big_cfg = dict(bg=BG, font=("Segoe UI", 13, "bold"))

        tk.Label(self, text="SESSION", bg=BG, fg=GOLD,
                 font=("Segoe UI", 9, "bold")).grid(row=0, column=0, columnspan=6, pady=(0, 6))

        # Victoires
        tk.Label(self, text="✔ Victoires", **lbl_cfg).grid(row=1, column=0, sticky="w")
        self._lbl_w = tk.Label(self, text="0", fg=X_COL, **big_cfg)
        self._lbl_w.grid(row=1, column=1, padx=(4, 16), sticky="w")
        self._bar_w = ttk.Progressbar(self, length=100, maximum=100, style="W.Horizontal.TProgressbar")
        self._bar_w.grid(row=1, column=2, padx=(0, 16))

        # Défaites
        tk.Label(self, text="✘ Défaites", **lbl_cfg).grid(row=1, column=3, sticky="w")
        self._lbl_l = tk.Label(self, text="0", fg=O_COL, **big_cfg)
        self._lbl_l.grid(row=1, column=4, padx=(4, 16), sticky="w")
        self._bar_l = ttk.Progressbar(self, length=100, maximum=100, style="L.Horizontal.TProgressbar")
        self._bar_l.grid(row=1, column=5, padx=(0, 16))

        # Nuls
        tk.Label(self, text="= Nuls", **lbl_cfg).grid(row=1, column=6, sticky="w")
        self._lbl_d = tk.Label(self, text="0", fg=GOLD, **big_cfg)
        self._lbl_d.grid(row=1, column=7, padx=(4, 16), sticky="w")
        self._bar_d = ttk.Progressbar(self, length=100, maximum=100, style="D.Horizontal.TProgressbar")
        self._bar_d.grid(row=1, column=8)

    def record(self, result: str):
        """result: 'win', 'loss', 'draw'"""
        if result == "win":
            self.wins += 1
        elif result == "loss":
            self.losses += 1
        else:
            self.draws += 1
        self._refresh()

    def _refresh(self):
        total = self.wins + self.losses + self.draws or 1
        self._lbl_w.config(text=str(self.wins))
        self._lbl_l.config(text=str(self.losses))
        self._lbl_d.config(text=str(self.draws))
        self._bar_w["value"] = self.wins / total * 100
        self._bar_l["value"] = self.losses / total * 100
        self._bar_d["value"] = self.draws / total * 100

    def reset(self):
        self.wins = self.losses = self.draws = 0
        self._refresh()


# ─────────────────────────────────────────────────────────────────────────────
# Widget : affichage vectoriel d'état
# ─────────────────────────────────────────────────────────────────────────────

class StateVectorPanel(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG3, padx=10, pady=8, **kwargs)
        tk.Label(self, text="Vecteur d'état (27,) float32", bg=BG3, fg=GOLD,
                 font=("Consolas", 10, "bold")).pack(anchor="w")
        self._txt_state = tk.Label(self, text="—", bg=BG3, fg=TXT,
                                   font=("Consolas", 9), justify="left", wraplength=540)
        self._txt_state.pack(anchor="w", pady=(2, 6))

        tk.Label(self, text="One-hot dernière action (9,)", bg=BG3, fg=GOLD_DIM,
                 font=("Consolas", 10, "bold")).pack(anchor="w")
        self._txt_action = tk.Label(self, text="—", bg=BG3, fg=TXT_DIM,
                                    font=("Consolas", 9), justify="left", wraplength=540)
        self._txt_action.pack(anchor="w")

    def update(self, state: np.ndarray, action: int = None):
        # State : afficher par groupes de 9 (joueur0 | joueur1 | vide)
        g0 = " ".join(f"{v:.0f}" for v in state[0:9])
        g1 = " ".join(f"{v:.0f}" for v in state[9:18])
        g2 = " ".join(f"{v:.0f}" for v in state[18:27])
        txt = f"P0 [{g0}]\nP1 [{g1}]\n∅  [{g2}]"
        self._txt_state.config(text=txt)

        if action is not None:
            oh = np.zeros(9, dtype=np.int8)
            oh[action] = 1
            oh_str = "[" + "  ".join(str(v) for v in oh) + "]"
            self._txt_action.config(text=oh_str)
        else:
            self._txt_action.config(text="—")

    def clear(self):
        self._txt_state.config(text="—")
        self._txt_action.config(text="—")


# ─────────────────────────────────────────────────────────────────────────────
# Onglet : Jeu (Human vs Random / Human vs Human / Random vs Random)
# ─────────────────────────────────────────────────────────────────────────────

class GameTab(tk.Frame):

    MODES = [
        "Human vs Random",
        "Human vs Human",
        "Agent vs Random",
        "Agent vs Human",
    ]


    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG, **kwargs)
        self._env = TicTacToe()
        self._mode = tk.StringVar(value=self.MODES[0])
        self._status_text = tk.StringVar(value="")
        self._auto_job = None  # after() id pour l'auto‑play
        self._game_over = False
        self._last_action = None
        self._current_agent_fn = None  # agent RL chargé
        self._agents = {}
        self._build_agents_list()


        self._build_ui()
        self._new_game()


    def _build_agents_list(self):
        base = "saved_models/tictactoe/"

        # Ajoute un mode random
        self._agents["Random (random)"] = (None, "Random (random)")

        # Agents DQN/DDQN/DDQNER/DDQNPER
        agents_config = [
            ("dqn",         DeepQLearning, ".pt"),
            ("ddqn",        DoubleDeepQLearning, ".pt"),
            ("ddqner",      DoubleDeepQLearningWithExperienceReplay, ".pt"),
            ("ddqnper",     DoubleDeepQLearningWithPrioritizedExperienceReplay, ".pt"),
        ]

        for agent_name, cls, suffix in agents_config:
            agent_folder = os.path.join(base, agent_name)
            if not os.path.exists(agent_folder):
                continue

            for fname in sorted(os.listdir(agent_folder)):
                if fname.endswith(suffix):
                    key = f"{agent_name}_{fname}"

                    def make_agent_fn(agent_name=agent_name, fname=fname, folder=agent_folder):
                        path = os.path.join(folder, fname)
                        if not os.path.exists(path):
                            print(f"❌ Fichier introuvable : {path}")
                            return None

                        agent = make_agent(agent_name, {})
                        data = torch.load(path, map_location="cpu")
                        data["epsilon"]       = 0.0
                        data["epsilon_min"]   = 0.0
                        data["epsilon_decay"] = 1.0
                        agent.load_state_dict(data, self._env.state_size, self._env.action_size)

                        def act(state):
                            return agent.select_action(self._env, greedy=True)
                        return act

                    self._agents[key] = (make_agent_fn, f"{agent_name} ({fname})")

        # TQL
        #tql_folder = os.path.join(base, "tql")
        tql_folder = os.path.join("saved_models", "tictactoe", "tql")
        if os.path.exists(tql_folder):
            for fname in sorted(os.listdir(tql_folder)):
                if fname.endswith(".pkl"):
                    key = f"tql_{fname}"
                    def make_tql_agent_fn(fname=fname, folder=tql_folder):
                        path = os.path.join(folder, fname)
                        if not os.path.exists(path):
                            print(f"❌ Fichier introuvable : {path}")
                            return None

                        agent = load_model("tql", "tictactoe", fname, seed=42, for_training=False)
                        if agent is None:
                            return None

                        def act(state):
                            return agent.select_action(self._env, greedy=True)
                        return act

                    self._agents[key] = (make_tql_agent_fn, f"TQL ({fname})")

        # Remplir le ComboBox dans l'UI
        values = list(self._agents.keys())
        if hasattr(self, "_agent_combo") and self._agent_combo:
            self._agent_combo["values"] = values
            if values:
                self._agent_combo.set(values[0])


    def _on_agent_select(self, event=None):
        key = self._agent_combo.get()
        loader, label_text = self._agents[key]
        if loader is None:
            self._current_agent_fn = None
            self._agent_label.config(text="Mode : Random")
        else:
            self._current_agent_fn = loader()
            self._agent_label.config(text=label_text)


    # ─────────────────────────────────────────────────────────────────────────────────────────
    # INTERFACE
    # ─────────────────────────────────────────────────────────────────────────────────────────


    def _build_ui(self):
        # ── En-tête : sélection de mode ─────────────────────────────────────
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=20, pady=(16, 0))

        tk.Label(top, text="TicTacToe RL", bg=BG, fg=GOLD,
                 font=("Segoe UI", 20, "bold")).pack(side="left")

        mode_frame = tk.Frame(top, bg=BG)
        mode_frame.pack(side="right")
        for mode in self.MODES:
            tk.Radiobutton(
                mode_frame, text=mode, variable=self._mode, value=mode,
                bg=BG, fg=TXT, selectcolor=BG2,
                activebackground=BG, activeforeground=GOLD,
                font=("Segoe UI", 10),
                command=self._on_mode_change
            ).pack(side="left", padx=8)

        # ── Corps central ───────────────────────────────────────────────────
        center = tk.Frame(self, bg=BG)
        center.pack(fill="both", expand=True, padx=20, pady=10)


        # Panneau gauche : plateau + status + bouton
        left = tk.Frame(center, bg=BG)
        left.pack(side="left", anchor="n")

        self._board_canvas = BoardCanvas(left, click_callback=self._on_cell_click)
        self._board_canvas.pack()

        self._status_lbl = tk.Label(left, textvariable=self._status_text,
                                    bg=BG, fg=GOLD, font=("Segoe UI", 13, "bold"),
                                    height=2, width=30, wraplength=340)
        self._status_lbl.pack(pady=8)

        btn = tk.Button(left, text="⟳  Nouvelle Partie",
                        bg=BG3, fg=GOLD, activebackground=GOLD, activeforeground=BG,
                        font=("Segoe UI", 11, "bold"), relief="flat", bd=0,
                        padx=18, pady=8, cursor="hand2",
                        command=self._new_game)
        btn.pack()


        # Panneau droit : vecteur état + stats
        right = tk.Frame(center, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(24, 0), anchor="n")

        self._vec_panel = StateVectorPanel(right)
        self._vec_panel.pack(fill="x", pady=(0, 16))

        tk.Label(right, text="Statistiques de session", bg=BG, fg=GOLD,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 6))
        self._stats = SessionStatsBar(right)
        self._stats.pack(anchor="w")

        tk.Button(right, text="Réinitialiser les stats",
                  bg=BG3, fg=TXT_DIM, font=("Segoe UI", 9), relief="flat",
                  cursor="hand2", command=self._stats.reset).pack(anchor="w", pady=(8, 0))


        # ── Choix d'agent ──
        agent_frame = tk.LabelFrame(
            right, text=" Agent (TicTacToe) ", bg=BG, fg=GOLD, font=("Segoe UI", 9), padx=10, pady=8
        )
        agent_frame.pack(fill="x", pady=(6, 0), anchor="w")

        self._agent_combo = ttk.Combobox(
            agent_frame,
            values=["Random (random)"],
            state="readonly",
        )
        self._agent_combo.pack(fill="x", padx=(0, 0))
        self._agent_combo.set("Random (random)")
        self._agent_combo.bind("<<ComboboxSelected>>", self._on_agent_select)

        self._agent_label = tk.Label(
            agent_frame,
            text="(Aucun agent sélectionné)",
            bg=BG, fg=TXT_DIM, font=("Segoe UI", 9),
        )
        self._agent_label.pack(anchor="w", pady=(4, 0))


    # ─────────────────────────────────────────────────────────────────────────────────────────
    # LOGIQUE DE JEU
    # ─────────────────────────────────────────────────────────────────────────────────────────


    def _on_mode_change(self):
        self._cancel_auto()
        mode = self._mode.get()
        if mode in ["Human vs Human", "Human vs Random", "Agent vs Human", "Agent vs Random"]:
            self._new_game()


    def _new_game(self):
        self._cancel_auto()
        self._env.reset()
        self._game_over = False
        self._last_action = None
        self._board_canvas.clear()
        self._board_canvas._draw_grid()
        self._vec_panel.clear()
        mode = self._mode.get()

        if mode == "Agent vs Random":
            self._status_text.set("Agent (X) joue…")
            self._schedule_agent_step()
        elif mode == "Agent vs Human":
            self._status_text.set("À vous de jouer ! (X)")
        elif mode == "Human vs Human":
            self._status_text.set("Tour du joueur X")
        else:  # Human vs Random
            self._status_text.set("À vous de jouer ! (X)")


    def _cancel_auto(self):
        if self._auto_job is not None:
            self.after_cancel(self._auto_job)
            self._auto_job = None


    def _refresh_display(self, winning_combo=None):
        state = self._env.get_state()
        self._board_canvas.render_board(self._env._board, winning_combo)
        self._vec_panel.update(state, self._last_action)


    # ─────────────────────────────────────────────────────────────────────────────────────────
    # MODES
    # ─────────────────────────────────────────────────────────────────────────────────────────


    def _on_cell_click(self, idx: int):
        mode = self._mode.get()
        if self._game_over:
            return
        if mode == "Agent vs Random":  # déjà géré par schedule_agent_step
            return

        if mode == "Human vs Human":
            self._hvh_play(idx)
        elif mode == "Agent vs Human":
            self._avia_play(idx)
        else:
            self._hvr_play(idx)  # Human vs Random


    # ── Mode Human vs Random ─────────────────────────────────────────────────

    def _hvr_play(self, idx: int):
        """Human (player 0 = X) joue, puis Random (player 1 = O)."""
        if self._env._current_player != 0:
            return
        if idx not in self._env.available_actions():
            return

        # Tour humain
        self._last_action = idx
        _, reward, done = self._env.step(idx)
        self._refresh_display()

        if done:
            self._end_game(reward, from_player=0)
            return

        # Tour random (O)
        self._status_text.set("O joue…")
        self.after(300, self._hvr_random_step)

    def _hvr_random_step(self):
        actions = self._env.available_actions()
        if not actions:
            return
        action = random.choice(actions)
        self._last_action = action
        _, reward, done = self._env.step(action)
        self._refresh_display()
        if done:
            self._end_game(-reward, from_player=1)
        else:
            self._status_text.set("À vous de jouer ! (X)")


    # ── Mode Human vs Human ─────────────────────────────────────────────────

    def _hvh_play(self, idx: int):
        """Écrit directement sur le board pour éviter env.step() qui change de tour."""
        if self._game_over:
            return
        env = self._env
        if env._board[idx] != 0:
            return  # case occupée

        player = env._current_player
        val = player + 1  # 1 ou 2
        env._board[idx] = val
        self._last_action = idx

        # Vérification victoire
        winning_combo = get_winning_combo(env._board, player)
        if winning_combo:
            env._done = True
            env._winner = player
            self._refresh_display(winning_combo=winning_combo)
            sym = "X" if player == 0 else "O"
            self._status_text.set(f"🏆 Joueur {sym} gagne !")
            result = "win" if player == 0 else "loss"
            self._stats.record(result)
            self._game_over = True
            return

        # Vérification nul
        if np.all(env._board != 0):
            env._done = True
            env._winner = None
            self._refresh_display()
            self._status_text.set("Match nul ! 🤝")
            self._stats.record("draw")
            self._game_over = True
            return

        # Changer de joueur
        env._current_player = 1 - player
        next_sym = "X" if env._current_player == 0 else "O"
        self._status_text.set(f"Tour du joueur {next_sym}")
        self._refresh_display()


    # ── Mode Agent vs Human (Agent = X, Human = O) ────────────────────────────

    def _avia_play(self, idx: int):
        if self._env._current_player != 1:
            return
        if idx not in self._env.available_actions():
            return

        # Humain joue (O)
        self._last_action = idx
        _, reward, done = self._env.step(idx)
        self._refresh_display()

        if done:
            self._end_game(-reward, from_player=1)
            return

        # Agent joue (X)
        self._status_text.set("Agent (X) joue…")
        self.after(300, self._avha_agent_step)

    def _avha_agent_step(self):
        state = self._env.get_state()
        if self._current_agent_fn is not None:
            action = self._current_agent_fn(state)
        else:
            actions = self._env.available_actions()
            action = random.choice(actions)
        self._last_action = action
        _, reward, done = self._env.step(action)
        self._refresh_display()
        if done:
            self._end_game(reward, from_player=0)
        else:
            self._status_text.set("À vous de jouer ! (O)")


    # ── Mode Agent vs Random (Agent = X, Random = O) ──────────────────────────

    def _schedule_agent_step(self):
        self._auto_job = self.after(250, self._agent_step)

    def _agent_step(self):
        self._auto_job = None
        if self._game_over:
            return
        actions = self._env.available_actions()
        if not actions:
            return
        if self._current_agent_fn is not None:
            action = self._current_agent_fn(self._env.get_state())
        else:
            action = random.choice(actions)
        self._last_action = action
        _, reward, done = self._env.step(action)
        self._refresh_display()
        if done:
            self._end_game(reward if self._env._current_player == 0 else -reward, from_player=self._env._current_player)
        else:
            self._schedule_agent_step()


    # ─────────────────────────────────────────────────────────────────────────────────────────
    # FIN DE PARTIE
    # ─────────────────────────────────────────────────────────────────────────────────────────

    def _end_game(self, reward_for_p0: float, from_player: int = 0):
        self._game_over = True
        env = self._env
        winning_combo = None

        if env._winner is not None:
            winning_combo = get_winning_combo(env._board, env._winner)
        self._board_canvas.render_board(env._board, winning_combo)

        if env._winner == 0:
            self._status_text.set("🏆 Agent (X) gagne !")
            self._stats.record("win")
        elif env._winner == 1:
            self._status_text.set("🏆 Humain (O) gagne !")
            self._stats.record("loss")
        else:
            self._status_text.set("Match nul ! 🤝")
            self._stats.record("draw")
# ─────────────────────────────────────────────────────────────────────────────
# Onglet : Simulation (benchmark)
# ─────────────────────────────────────────────────────────────────────────────

class SimulationTab(tk.Frame):

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG, **kwargs)
        self._running = False
        self._thread = None
        self._build_ui()

    def _build_ui(self):
        pad = dict(padx=20, pady=10)

        # Titre
        tk.Label(self, text="Benchmark — Random vs Random", bg=BG, fg=GOLD,
                 font=("Segoe UI", 18, "bold")).pack(anchor="w", **pad)

        # Paramètres
        cfg_frame = tk.Frame(self, bg=BG2, padx=16, pady=12)
        cfg_frame.pack(fill="x", padx=20, pady=(0, 10))

        tk.Label(cfg_frame, text="Nombre de parties :", bg=BG2, fg=TXT,
                 font=("Segoe UI", 11)).grid(row=0, column=0, sticky="w")
        self._n_var = tk.IntVar(value=10000)
        tk.Spinbox(cfg_frame, from_=100, to=1000000, increment=1000,
                   textvariable=self._n_var, width=10,
                   bg=BG3, fg=TXT, buttonbackground=BG3,
                   font=("Segoe UI", 11), relief="flat").grid(row=0, column=1, padx=10)

        self._btn_start = tk.Button(
            cfg_frame, text="▶  Lancer",
            bg=GOLD, fg=BG, activebackground=GOLD_DIM, activeforeground=BG,
            font=("Segoe UI", 11, "bold"), relief="flat", padx=16, pady=6,
            cursor="hand2", command=self._start
        )
        self._btn_start.grid(row=0, column=2, padx=10)

        self._btn_stop = tk.Button(
            cfg_frame, text="■  Arrêter",
            bg=BG3, fg=TXT, font=("Segoe UI", 11), relief="flat", padx=16, pady=6,
            cursor="hand2", state="disabled", command=self._stop
        )
        self._btn_stop.grid(row=0, column=3)

        # Barre de progression
        self._progress = ttk.Progressbar(self, length=700, maximum=100,
                                         style="Sim.Horizontal.TProgressbar")
        self._progress.pack(padx=20, pady=(0, 4))
        self._prog_lbl = tk.Label(self, text="", bg=BG, fg=TXT_DIM, font=("Segoe UI", 9))
        self._prog_lbl.pack()

        # Résultats
        res_frame = tk.Frame(self, bg=BG2, padx=20, pady=16)
        res_frame.pack(fill="x", padx=20, pady=10)

        def big_stat(parent, label, color, col):
            f = tk.Frame(parent, bg=BG2)
            f.grid(row=0, column=col, padx=24)
            tk.Label(f, text=label, bg=BG2, fg=TXT_DIM,
                     font=("Segoe UI", 9)).pack()
            lbl = tk.Label(f, text="—", bg=BG2, fg=color,
                           font=("Segoe UI", 24, "bold"))
            lbl.pack()
            return lbl

        self._lbl_games   = big_stat(res_frame, "Parties",    TXT,    0)
        self._lbl_speed   = big_stat(res_frame, "Parties/sec", GOLD,  1)
        self._lbl_winx    = big_stat(res_frame, "% X gagne",  X_COL,  2)
        self._lbl_wino    = big_stat(res_frame, "% O gagne",  O_COL,  3)
        self._lbl_draws   = big_stat(res_frame, "% Nuls",     GOLD,   4)

        # Log
        log_frame = tk.Frame(self, bg=BG2)
        log_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        self._log = tk.Text(log_frame, bg=BG2, fg=TXT_DIM,
                            font=("Consolas", 9), relief="flat",
                            height=8, state="disabled")
        scroll = ttk.Scrollbar(log_frame, command=self._log.yview)
        self._log.configure(yscrollcommand=scroll.set)
        self._log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def _log_append(self, msg: str):
        self._log.config(state="normal")
        self._log.insert("end", msg + "\n")
        self._log.see("end")
        self._log.config(state="disabled")

    def _start(self):
        if self._running:
            return
        self._running = True
        self._btn_start.config(state="disabled")
        self._btn_stop.config(state="normal")
        self._progress["value"] = 0
        n = self._n_var.get()
        self._log_append(f"[{time.strftime('%H:%M:%S')}] Lancement de {n} parties…")
        self._thread = threading.Thread(target=self._run_sim, args=(n,), daemon=True)
        self._thread.start()

    def _stop(self):
        self._running = False

    def _run_sim(self, n: int):
        env = TicTacToe()
        wins_x = 0
        wins_o = 0
        draws  = 0
        t0 = time.perf_counter()
        update_every = max(1, n // 200)

        for i in range(n):
            if not self._running:
                break
            env.reset()
            while not env.is_game_over():
                actions = env.available_actions()
                env.step(random.choice(actions))
            if env._winner == 0:
                wins_x += 1
            elif env._winner == 1:
                wins_o += 1
            else:
                draws += 1

            if (i + 1) % update_every == 0:
                elapsed = time.perf_counter() - t0
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                pct = (i + 1) / n * 100
                self.after(0, self._update_ui, i + 1, speed, wins_x, wins_o, draws, pct)

        elapsed = time.perf_counter() - t0
        total = wins_x + wins_o + draws
        speed = total / elapsed if elapsed > 0 else 0
        self.after(0, self._finish_ui, total, speed, wins_x, wins_o, draws, elapsed)

    def _update_ui(self, games, speed, wins_x, wins_o, draws, pct):
        if not self._running:
            return
        total = wins_x + wins_o + draws or 1
        self._progress["value"] = pct
        self._prog_lbl.config(text=f"{games:,} / {self._n_var.get():,} parties")
        self._lbl_games.config(text=f"{games:,}")
        self._lbl_speed.config(text=f"{speed:,.0f}")
        self._lbl_winx.config(text=f"{wins_x/total*100:.1f}%")
        self._lbl_wino.config(text=f"{wins_o/total*100:.1f}%")
        self._lbl_draws.config(text=f"{draws/total*100:.1f}%")

    def _finish_ui(self, total, speed, wins_x, wins_o, draws, elapsed):
        self._running = False
        self._btn_start.config(state="normal")
        self._btn_stop.config(state="disabled")
        self._progress["value"] = 100
        self._prog_lbl.config(text=f"{total:,} parties terminées en {elapsed:.2f}s")
        self._update_ui(total, speed, wins_x, wins_o, draws, 100)
        self._log_append(
            f"[{time.strftime('%H:%M:%S')}] Terminé — {total:,} parties | "
            f"{speed:,.0f} p/s | X={wins_x/total*100:.1f}% | "
            f"O={wins_o/total*100:.1f}% | ={draws/total*100:.1f}%"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Application principale
# ─────────────────────────────────────────────────────────────────────────────

class TicTacToeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TicTacToe — Reinforcement Learning")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(860, 620)

        self._setup_styles()
        self._build_notebook()

    def _build_agents_list(self):
        base = "saved_models/tictactoe/"

        # Ajoute un mode random
        self._agents["Random (random)"] = (None, "Random (random)")

        # Agents DQN/DDQN/DDQNER/DDQNPER
        agents_config = [
            ("dqn",         DeepQLearning, ".pt"),
            ("ddqn",        DoubleDeepQLearning, ".pt"),
            ("ddqner",      DoubleDeepQLearningWithExperienceReplay, ".pt"),
            ("ddqnper",     DoubleDeepQLearningWithPrioritizedExperienceReplay, ".pt"),
        ]

        for agent_name, cls, suffix in agents_config:
            folder = os.path.join(base, agent_name)
            if not os.path.exists(folder):
                continue

            for fname in sorted(os.listdir(folder)):
                if fname.endswith(suffix):
                    key = f"{agent_name}_{fname}"

                    def make_agent_fn(agent_name=agent_name, fname=fname, folder=folder):
                        path = os.path.join(folder, fname)
                        if not os.path.exists(path):
                            print(f"Fichier introuvable : {path}")
                            return None

                        agent = make_agent(agent_name, {})
                        data = torch.load(path, map_location="cpu")
                        data["epsilon"]       = 0.0
                        data["epsilon_min"]   = 0.0
                        data["epsilon_decay"] = 1.0
                        agent.load_state_dict(data, self._env.state_size, self._env.action_size)

                        def act(state):
                            return agent.select_action(self._env, greedy=True)
                        return act

                    self._agents[key] = (make_agent_fn, f"{agent_name} ({fname})")

        # TQL
        tql_folder = os.path.join(base, "tql")
        if os.path.exists(tql_folder):
            for fname in sorted(os.listdir(tql_folder)):
                if fname.endswith(".pkl"):
                    key = f"tql_{fname}"
                    def make_tql_agent_fn(fname=fname, folder=tql_folder):
                        path = os.path.join(folder, fname)
                        if not os.path.exists(path):
                            print(f"Fichier introuvable : {path}")
                            return None

                        agent = load_model("tql", "tictactoe", fname, seed=42, for_training=False)
                        if agent is None:
                            return None

                        def act(state):
                            return agent.select_action(self._env, greedy=True)
                        return act

                    self._agents[key] = (make_tql_agent_fn, f"TQL ({fname})")

        # Remplir le ComboBox dans l'UI
        self._agent_combo["values"] = list(self._agents.keys())

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG2, foreground=TXT,
                        padding=[16, 8], font=("Segoe UI", 11))
        style.map("TNotebook.Tab",
                  background=[("selected", BG3)],
                  foreground=[("selected", GOLD)])

        # Barres de stats
        for name, color in [("W", X_COL), ("L", O_COL), ("D", GOLD), ("Sim", GOLD)]:
            style.configure(f"{name}.Horizontal.TProgressbar",
                            troughcolor=BG3, background=color,
                            thickness=8, borderwidth=0)

        style.configure("TScrollbar", background=BG3, troughcolor=BG2, borderwidth=0)
        style.configure("TSpinbox", fieldbackground=BG3, background=BG3, foreground=TXT)

    def _build_notebook(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=0, pady=0)

        game_tab = GameTab(nb)
        sim_tab  = SimulationTab(nb)

        nb.add(game_tab, text="  🎮  Jeu  ")
        nb.add(sim_tab,  text="  📊  Simulation  ")

    def _on_agent_select(self, event=None):
        key = self._agent_combo.get()
        loader, label_text = self._agents[key]
        if loader is None:
            self._current_agent_fn = None
            self._agent_label.config(text="Mode : Random")
        else:
            self._current_agent_fn = loader()
            self._agent_label.config(text=label_text)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = TicTacToeApp()
    app.mainloop()
