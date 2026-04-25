"""
gui/tictactoe_gui.py
====================
Interface graphique tkinter pour l'environnement TicTacToe.

Modes disponibles :
  • Human vs Random  — l'humain joue X, l'IA joue O au hasard
  • Human vs Human   — 2 joueurs sur le même clavier, alternance X/O
  • Agent vs Random  — l'agent RL joue X, Random joue O (auto-play)
  • Agent vs Human   — l'agent RL joue X, l'humain joue O

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

import torch

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


# ─────────────────────────────────────────────────────────────────────────────
# Chargement générique d'agents
# ─────────────────────────────────────────────────────────────────────────────

# Mapping dossier → (module, classe, interface)
# interface : "rl" = select_action(env, greedy=True)
#             "act" = .act(env)
AGENT_REGISTRY = {
    "dqn":        ("agents.dqn",    "DeepQLearning",                             "rl"),
    "ddqn":       ("agents.ddqn",   "DoubleDeepQLearning",                       "rl"),
    "ddqner":     ("agents.ddqner", "DoubleDeepQLearningWithExperienceReplay",    "rl"),
    "ddqnper":    ("agents.ddqnper","DoubleDeepQLearningWithPrioritizedExperienceReplay", "rl"),
    "tql":        ("agents.tabular_q_learning", "TabularQLearning",              "rl"),
    "apprentice": ("agents.expert_apprentice", "ExpertApprenticeAgent",                 "act"),
    "ppo":        ("agents.ppo",    "PPOAgent",                                  "act"),
}


def load_agent_fn(agent_type: str, model_path: str, env: TicTacToe):
    """
    Retourne une fonction act(env) -> action pour n'importe quel agent.
    Gère deux interfaces :
      - "rl"  : agents DQN/DDQN/TQL  → agent.select_action(env, greedy=True)
      - "act" : ExpertApprentice/PPO  → agent.act(env)
    """
    if agent_type not in AGENT_REGISTRY:
        print(f"⚠️  Type d'agent inconnu : {agent_type}")
        return None

    module_name, class_name, interface = AGENT_REGISTRY[agent_type]

    try:
        import importlib
        module = importlib.import_module(module_name)
        cls    = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        print(f"❌ Impossible d'importer {class_name} depuis {module_name} : {e}")
        return None

    try:
        if interface == "act":
            # ExpertApprenticeAgent / PPOAgent : constructeur (state_size, action_size, model_path)
            agent = cls(env.state_size, env.action_size, model_path=model_path)
            def act_fn(env_ref):
                return agent.act(env_ref, render=True)
            return act_fn

        elif interface == "rl":
            if agent_type == "tql":
                import pickle
                with open(model_path, "rb") as f:
                    data = pickle.load(f)
                agent = cls(
                    alpha         = data["alpha"],
                    gamma         = data["gamma"],
                    epsilon       = 0.0,
                    epsilon_min   = 0.0,
                    epsilon_decay = 1.0,
                )
                agent._q_table = data["q_table"]
            else:
                data = torch.load(model_path, map_location="cpu")
                data["epsilon"]       = 0.0
                data["epsilon_min"]   = 0.0
                data["epsilon_decay"] = 1.0
                agent = cls.from_config({})
                agent.load_state_dict(data, env.state_size, env.action_size)

            def rl_fn(env_ref):
                return agent.select_action(env_ref, greedy=True)
            return rl_fn

    except Exception as e:
        print(f"❌ Erreur lors du chargement de {model_path} : {e}")
        return None


def scan_saved_models(base_dir: str, env: TicTacToe) -> dict:
    """
    Parcourt base_dir et retourne un dict :
      label -> callable act(env) -> int
    Gère automatiquement tous les sous-dossiers connus dans AGENT_REGISTRY.
    """
    agents = {"🎲 Random": None}

    if not os.path.isdir(base_dir):
        print(f"⚠️  Dossier introuvable : {base_dir}")
        return agents

    for folder_name in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        agent_type = folder_name.lower()
        ext = ".pkl" if agent_type == "tql" else ".pt"

        for fname in sorted(os.listdir(folder_path)):
            if not fname.endswith(ext):
                continue
            model_path = os.path.join(folder_path, fname)
            label      = f"{folder_name} › {fname}"

            # Capture par valeur avec default args
            def make_fn(t=agent_type, p=model_path):
                return load_agent_fn(t, p, env)

            agents[label] = make_fn

    return agents


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_winning_combo(board: np.ndarray, player: int):
    val = player + 1
    for combo in TicTacToe._WINNING_COMBOS:
        if all(board[i] == val for i in combo):
            return combo
    return None


def cell_center(idx: int):
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
        for col in range(1, 3):
            x = p + col * CELL_SIZE
            self.create_line(x, p, x, s - p, fill=BG3, width=3, tags="grid")
        for row in range(1, 3):
            y = p + row * CELL_SIZE
            self.create_line(p, y, s - p, y, fill=BG3, width=3, tags="grid")

    def _on_click(self, event):
        if self._click_cb is None:
            return
        col = (event.x - PADDING) // CELL_SIZE
        row = (event.y - PADDING) // CELL_SIZE
        if 0 <= row < 3 and 0 <= col < 3:
            self._click_cb(row * 3 + col)

    def render_board(self, board: np.ndarray, winning_combo=None):
        self.delete("piece", "winline")
        for idx in range(9):
            val = board[idx]
            if val == 0:
                continue
            cx, cy = cell_center(idx)
            color  = X_COL if val == 1 else O_COL
            r      = CELL_SIZE // 2 - 20
            if val == 1:  # X
                self.create_line(cx-r, cy-r, cx+r, cy+r,
                                 fill=color, width=6, capstyle=tk.ROUND, tags="piece")
                self.create_line(cx+r, cy-r, cx-r, cy+r,
                                 fill=color, width=6, capstyle=tk.ROUND, tags="piece")
            else:  # O
                self.create_oval(cx-r, cy-r, cx+r, cy+r,
                                 outline=color, width=6, tags="piece")

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
        self.wins = self.losses = self.draws = 0

        lbl = dict(bg=BG, fg=TXT, font=("Segoe UI", 10))
        big = dict(bg=BG, font=("Segoe UI", 13, "bold"))

        tk.Label(self, text="SESSION", bg=BG, fg=GOLD,
                 font=("Segoe UI", 9, "bold")).grid(row=0, column=0, columnspan=9, pady=(0, 6))

        tk.Label(self, text="✔ Victoires", **lbl).grid(row=1, column=0, sticky="w")
        self._lbl_w = tk.Label(self, text="0", fg=X_COL, **big)
        self._lbl_w.grid(row=1, column=1, padx=(4, 16), sticky="w")
        self._bar_w = ttk.Progressbar(self, length=100, maximum=100, style="W.Horizontal.TProgressbar")
        self._bar_w.grid(row=1, column=2, padx=(0, 16))

        tk.Label(self, text="✘ Défaites", **lbl).grid(row=1, column=3, sticky="w")
        self._lbl_l = tk.Label(self, text="0", fg=O_COL, **big)
        self._lbl_l.grid(row=1, column=4, padx=(4, 16), sticky="w")
        self._bar_l = ttk.Progressbar(self, length=100, maximum=100, style="L.Horizontal.TProgressbar")
        self._bar_l.grid(row=1, column=5, padx=(0, 16))

        tk.Label(self, text="= Nuls", **lbl).grid(row=1, column=6, sticky="w")
        self._lbl_d = tk.Label(self, text="0", fg=GOLD, **big)
        self._lbl_d.grid(row=1, column=7, padx=(4, 16), sticky="w")
        self._bar_d = ttk.Progressbar(self, length=100, maximum=100, style="D.Horizontal.TProgressbar")
        self._bar_d.grid(row=1, column=8)

    def record(self, result: str):
        if result == "win":   self.wins   += 1
        elif result == "loss": self.losses += 1
        else:                  self.draws  += 1
        self._refresh()

    def _refresh(self):
        total = self.wins + self.losses + self.draws or 1
        self._lbl_w.config(text=str(self.wins))
        self._lbl_l.config(text=str(self.losses))
        self._lbl_d.config(text=str(self.draws))
        self._bar_w["value"] = self.wins   / total * 100
        self._bar_l["value"] = self.losses / total * 100
        self._bar_d["value"] = self.draws  / total * 100

    def reset(self):
        self.wins = self.losses = self.draws = 0
        self._refresh()


# ─────────────────────────────────────────────────────────────────────────────
# Widget : vecteur d'état
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
        g0 = " ".join(f"{v:.0f}" for v in state[0:9])
        g1 = " ".join(f"{v:.0f}" for v in state[9:18])
        g2 = " ".join(f"{v:.0f}" for v in state[18:27])
        self._txt_state.config(text=f"P0 [{g0}]\nP1 [{g1}]\n∅  [{g2}]")

        if action is not None:
            oh = np.zeros(9, dtype=np.int8)
            oh[action] = 1
            self._txt_action.config(text="[" + "  ".join(str(v) for v in oh) + "]")
        else:
            self._txt_action.config(text="—")

    def clear(self):
        self._txt_state.config(text="—")
        self._txt_action.config(text="—")


# ─────────────────────────────────────────────────────────────────────────────
# Onglet Jeu
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

        # Environnement interne (utilisé pour le board réel)
        self._env       = TicTacToe()
        # Board manuel pour Human vs Human (env ne supporte pas 2 humains)
        self._hvh_board = np.zeros(9, dtype=np.int8)
        self._hvh_player = 0  # 0 = X, 1 = O

        self._mode        = tk.StringVar(value=self.MODES[0])
        self._status_text = tk.StringVar(value="")
        self._auto_job    = None
        self._game_over   = False
        self._last_action = None

        # Catalogue d'agents : label -> loader ou None (Random)
        base = os.path.join(ROOT, "modeles_gui", "tictactoe")
        self._agent_catalog = scan_saved_models(base, self._env)
        self._current_act_fn = None  # callable act(env)->int ou None

        self._build_ui()
        self._populate_agent_combo()
        self._new_game()

    # ─────────────────────────────────────────────────────────────────────────
    # Construction de l'UI
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # En-tête
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

        # Corps
        center = tk.Frame(self, bg=BG)
        center.pack(fill="both", expand=True, padx=20, pady=10)

        # Gauche : plateau
        left = tk.Frame(center, bg=BG)
        left.pack(side="left", anchor="n")

        self._board_canvas = BoardCanvas(left, click_callback=self._on_cell_click)
        self._board_canvas.pack()

        self._status_lbl = tk.Label(
            left, textvariable=self._status_text,
            bg=BG, fg=GOLD, font=("Segoe UI", 13, "bold"),
            height=2, width=30, wraplength=340
        )
        self._status_lbl.pack(pady=8)

        tk.Button(
            left, text="⟳  Nouvelle Partie",
            bg=BG3, fg=GOLD, activebackground=GOLD, activeforeground=BG,
            font=("Segoe UI", 11, "bold"), relief="flat", bd=0,
            padx=18, pady=8, cursor="hand2",
            command=self._new_game
        ).pack()

        # Droite : vecteur état + stats + choix agent
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
                  cursor="hand2", command=self._stats.reset
                  ).pack(anchor="w", pady=(8, 0))

        # Sélection d'agent
        agent_frame = tk.LabelFrame(
            right, text=" Agent chargé ", bg=BG, fg=GOLD,
            font=("Segoe UI", 9), padx=10, pady=8
        )
        agent_frame.pack(fill="x", pady=(16, 0), anchor="w")

        self._agent_combo = ttk.Combobox(agent_frame, state="readonly")
        self._agent_combo.pack(fill="x")
        self._agent_combo.bind("<<ComboboxSelected>>", self._on_agent_select)

        self._agent_status_lbl = tk.Label(
            agent_frame, text="Aucun agent chargé",
            bg=BG, fg=TXT_DIM, font=("Segoe UI", 9)
        )
        self._agent_status_lbl.pack(anchor="w", pady=(4, 0))

    def _populate_agent_combo(self):
        labels = list(self._agent_catalog.keys())
        self._agent_combo["values"] = labels
        if labels:
            self._agent_combo.set(labels[0])
            self._on_agent_select()

    # ─────────────────────────────────────────────────────────────────────────
    # Gestion du mode et de l'agent
    # ─────────────────────────────────────────────────────────────────────────

    def _on_mode_change(self):
        self._cancel_auto()
        self._new_game()

    def _on_agent_select(self, event=None):
        label  = self._agent_combo.get()
        loader = self._agent_catalog.get(label)

        if loader is None:
            # Random
            self._current_act_fn = None
            self._agent_status_lbl.config(text="Mode : Random", fg=TXT_DIM)
        else:
            self._agent_status_lbl.config(text="⏳ Chargement…", fg=GOLD_DIM)
            self.update_idletasks()
            fn = loader()
            if fn is None:
                self._current_act_fn = None
                self._agent_status_lbl.config(text="❌ Échec du chargement", fg=X_COL)
            else:
                self._current_act_fn = fn
                self._agent_status_lbl.config(text=f"✅ {label}", fg=GOLD)

    # ─────────────────────────────────────────────────────────────────────────
    # Gestion du jeu
    # ─────────────────────────────────────────────────────────────────────────

    def _new_game(self):
        self._cancel_auto()
        self._game_over   = False
        self._last_action = None

        mode = self._mode.get()

        if mode == "Human vs Human":
            # On gère le board manuellement, pas besoin de l'env TicTacToe
            self._hvh_board  = np.zeros(9, dtype=np.int8)
            self._hvh_player = 0
            self._board_canvas.clear()
            self._board_canvas._draw_grid()
            self._vec_panel.clear()
            self._status_text.set("Tour du joueur X")

        else:
            self._env.reset()
            self._board_canvas.clear()
            self._board_canvas._draw_grid()
            self._vec_panel.clear()

            if mode == "Agent vs Random":
                self._status_text.set("Agent (X) réfléchit…")
                self._schedule_agent_step()
            elif mode == "Agent vs Human":
                self._status_text.set("Agent (X) réfléchit…")
                self._schedule_agent_step()
            else:  # Human vs Random
                self._status_text.set("À vous de jouer ! (X)")

    def _cancel_auto(self):
        if self._auto_job is not None:
            self.after_cancel(self._auto_job)
            self._auto_job = None

    def _refresh_display(self, board: np.ndarray, winning_combo=None):
        self._board_canvas.render_board(board, winning_combo)
        state = self._env.get_state() if self._mode.get() != "Human vs Human" else np.zeros(27)
        self._vec_panel.update(state, self._last_action)

    # ─────────────────────────────────────────────────────────────────────────
    # Dispatch des clics
    # ─────────────────────────────────────────────────────────────────────────

    def _on_cell_click(self, idx: int):
        if self._game_over:
            return
        mode = self._mode.get()

        if mode == "Human vs Human":
            self._hvh_play(idx)
        elif mode == "Human vs Random":
            self._hvr_human_play(idx)
        elif mode == "Agent vs Human":
            self._avh_human_play(idx)
        # "Agent vs Random" : pas de clic humain

    # ─────────────────────────────────────────────────────────────────────────
    # Mode Human vs Human  (board manuel, pas d'env.step)
    # ─────────────────────────────────────────────────────────────────────────

    def _hvh_play(self, idx: int):
        if self._hvh_board[idx] != 0:
            return

        player = self._hvh_player
        self._hvh_board[idx] = player + 1  # 1=X, 2=O
        self._last_action     = idx

        winning_combo = get_winning_combo(self._hvh_board, player)
        if winning_combo:
            self._board_canvas.render_board(self._hvh_board, winning_combo)
            sym = "X" if player == 0 else "O"
            self._status_text.set(f"🏆 Joueur {sym} gagne !")
            self._stats.record("win" if player == 0 else "loss")
            self._game_over = True
            return

        if np.all(self._hvh_board != 0):
            self._board_canvas.render_board(self._hvh_board)
            self._status_text.set("Match nul ! 🤝")
            self._stats.record("draw")
            self._game_over = True
            return

        self._hvh_player = 1 - player
        sym = "X" if self._hvh_player == 0 else "O"
        self._status_text.set(f"Tour du joueur {sym}")
        self._board_canvas.render_board(self._hvh_board)

    # ─────────────────────────────────────────────────────────────────────────
    # Mode Human vs Random  (humain = X / valeur 1, env gère tout)
    # ─────────────────────────────────────────────────────────────────────────

    def _hvr_human_play(self, idx: int):
        if idx not in self._env.available_actions():
            return

        self._last_action = idx
        # env.step joue X ET fait jouer O (Random) dans la foulée si la partie continue
        _, reward, done = self._env.step(idx)
        self._refresh_display(self._env._board)

        if done:
            self._end_game()
        else:
            self._status_text.set("À vous de jouer ! (X)")

    # ─────────────────────────────────────────────────────────────────────────
    # Mode Agent vs Random  (agent = X, Random = O intégré dans env.step)
    # ─────────────────────────────────────────────────────────────────────────

    def _schedule_agent_step(self):
        self._auto_job = self.after(400, self._agent_step)

    def _agent_step(self):
        self._auto_job = None
        if self._game_over:
            return

        action = self._pick_agent_action()
        if action is None:
            return

        self._last_action = action
        # env.step joue X et fait jouer Random (O) en interne
        _, reward, done = self._env.step(action)
        self._refresh_display(self._env._board)

        if done:
            self._end_game()
        else:
            if self._mode.get() == "Agent vs Random":
                self._schedule_agent_step()
            else:
                # Agent vs Human : maintenant c'est au tour de l'humain
                self._status_text.set("À vous de jouer ! (O)")

    # ─────────────────────────────────────────────────────────────────────────
    # Mode Agent vs Human
    # L'agent joue X via env.step (ce qui inclut le Random interne).
    # MAIS on veut que l'humain joue O — problème : env.step fait jouer
    # Random après X automatiquement.
    # Solution : on manipule le board directement pour le coup de l'humain,
    # exactement comme en HvH, puis on vérifie les conditions de fin.
    # ─────────────────────────────────────────────────────────────────────────

    def _avh_human_play(self, idx: int):
        """L'humain joue O (valeur 2) directement sur le board."""
        if idx not in self._env.available_actions():
            return

        # Écriture directe (O = valeur 2)
        self._env._board[idx] = 2
        self._last_action = idx
        self._refresh_display(self._env._board)

        # Vérifie si l'humain a gagné
        winning_combo = get_winning_combo(self._env._board, 1)  # joueur 1 = O
        if winning_combo:
            self._env._done   = True
            self._env._winner = 1
            self._board_canvas.render_board(self._env._board, winning_combo)
            self._status_text.set("🏆 Humain (O) gagne !")
            self._stats.record("loss")
            self._game_over = True
            return

        # Vérifie le match nul
        if np.all(self._env._board != 0):
            self._env._done   = True
            self._env._winner = None
            self._status_text.set("Match nul ! 🤝")
            self._stats.record("draw")
            self._game_over = True
            return

        # Tour de l'agent (X)
        self._status_text.set("Agent (X) réfléchit…")
        self._schedule_agent_step()

    # ─────────────────────────────────────────────────────────────────────────
    # Sélection de l'action de l'agent
    # ─────────────────────────────────────────────────────────────────────────

    def _pick_agent_action(self) -> int:
        actions = self._env.available_actions()
        if not actions:
            return None
        if self._current_act_fn is not None:
            try:
                return self._current_act_fn(self._env)
            except Exception as e:
                print(f"⚠️  Erreur agent : {e} — fallback Random")
        return random.choice(actions)

    # ─────────────────────────────────────────────────────────────────────────
    # Fin de partie (pour les modes utilisant env.step)
    # ─────────────────────────────────────────────────────────────────────────

    def _end_game(self):
        self._game_over = True
        env = self._env
        winning_combo = None

        if env._winner is not None:
            winning_combo = get_winning_combo(env._board, env._winner)
        self._board_canvas.render_board(env._board, winning_combo)

        if env._winner == 0:
            self._status_text.set("🏆 Agent / X gagne !")
            self._stats.record("win")
        elif env._winner == 1:
            self._status_text.set("🏆 O gagne !")
            self._stats.record("loss")
        else:
            self._status_text.set("Match nul ! 🤝")
            self._stats.record("draw")


# ─────────────────────────────────────────────────────────────────────────────
# Onglet Simulation
# ─────────────────────────────────────────────────────────────────────────────

class SimulationTab(tk.Frame):

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG, **kwargs)
        self._running = False
        self._thread  = None

        # Catalogue d'agents pour la simulation
        base = os.path.join(ROOT, "modeles_gui", "tictactoe")
        self._env_sim     = TicTacToe()
        self._agent_catalog = scan_saved_models(base, self._env_sim)
        self._sim_act_fn  = None  # agent chargé pour la simulation

        self._build_ui()

    def _build_ui(self):
        pad = dict(padx=20, pady=10)

        tk.Label(self, text="Benchmark — Agent vs Random", bg=BG, fg=GOLD,
                 font=("Segoe UI", 18, "bold")).pack(anchor="w", **pad)

        # Config
        cfg = tk.Frame(self, bg=BG2, padx=16, pady=12)
        cfg.pack(fill="x", padx=20, pady=(0, 10))

        # Nombre de parties
        tk.Label(cfg, text="Parties :", bg=BG2, fg=TXT,
                 font=("Segoe UI", 11)).grid(row=0, column=0, sticky="w")
        self._n_var = tk.IntVar(value=10000)
        tk.Spinbox(cfg, from_=100, to=1_000_000, increment=1000,
                   textvariable=self._n_var, width=10,
                   bg=BG3, fg=TXT, buttonbackground=BG3,
                   font=("Segoe UI", 11), relief="flat").grid(row=0, column=1, padx=10)

        # Sélection agent
        tk.Label(cfg, text="Agent :", bg=BG2, fg=TXT,
                 font=("Segoe UI", 11)).grid(row=0, column=2, sticky="w", padx=(16, 0))
        self._sim_combo = ttk.Combobox(cfg, values=list(self._agent_catalog.keys()),
                                       state="readonly", width=30)
        self._sim_combo.grid(row=0, column=3, padx=8)
        labels = list(self._agent_catalog.keys())
        if labels:
            self._sim_combo.set(labels[0])
        self._sim_combo.bind("<<ComboboxSelected>>", self._on_sim_agent_select)

        self._btn_start = tk.Button(
            cfg, text="▶  Lancer",
            bg=GOLD, fg=BG, activebackground=GOLD_DIM, activeforeground=BG,
            font=("Segoe UI", 11, "bold"), relief="flat", padx=16, pady=6,
            cursor="hand2", command=self._start
        )
        self._btn_start.grid(row=0, column=4, padx=10)

        self._btn_stop = tk.Button(
            cfg, text="■  Arrêter",
            bg=BG3, fg=TXT, font=("Segoe UI", 11), relief="flat", padx=16, pady=6,
            cursor="hand2", state="disabled", command=self._stop
        )
        self._btn_stop.grid(row=0, column=5)

        self._sim_status_lbl = tk.Label(cfg, text="", bg=BG2, fg=TXT_DIM,
                                         font=("Segoe UI", 9))
        self._sim_status_lbl.grid(row=1, column=2, columnspan=4, sticky="w", pady=(6, 0))

        # Progression
        self._progress = ttk.Progressbar(self, length=700, maximum=100,
                                          style="Sim.Horizontal.TProgressbar")
        self._progress.pack(padx=20, pady=(0, 4))
        self._prog_lbl = tk.Label(self, text="", bg=BG, fg=TXT_DIM, font=("Segoe UI", 9))
        self._prog_lbl.pack()

        # Résultats
        res = tk.Frame(self, bg=BG2, padx=20, pady=16)
        res.pack(fill="x", padx=20, pady=10)

        def big(parent, label, color, col):
            f = tk.Frame(parent, bg=BG2)
            f.grid(row=0, column=col, padx=24)
            tk.Label(f, text=label, bg=BG2, fg=TXT_DIM,
                     font=("Segoe UI", 9)).pack()
            lbl = tk.Label(f, text="—", bg=BG2, fg=color,
                           font=("Segoe UI", 24, "bold"))
            lbl.pack()
            return lbl

        self._lbl_games = big(res, "Parties",     TXT,   0)
        self._lbl_speed = big(res, "Parties/sec", GOLD,  1)
        self._lbl_winx  = big(res, "% X gagne",  X_COL, 2)
        self._lbl_wino  = big(res, "% O gagne",  O_COL, 3)
        self._lbl_draws = big(res, "% Nuls",     GOLD,  4)

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

    def _on_sim_agent_select(self, event=None):
        label  = self._sim_combo.get()
        loader = self._agent_catalog.get(label)
        if loader is None:
            self._sim_act_fn = None
            self._sim_status_lbl.config(text="Mode : Random", fg=TXT_DIM)
        else:
            self._sim_status_lbl.config(text="⏳ Chargement…", fg=GOLD_DIM)
            self.update_idletasks()
            fn = loader()
            if fn is None:
                self._sim_act_fn = None
                self._sim_status_lbl.config(text="❌ Échec du chargement", fg=X_COL)
            else:
                self._sim_act_fn = fn
                self._sim_status_lbl.config(text=f"✅ {label}", fg=GOLD)

    def _log_append(self, msg: str):
        self._log.config(state="normal")
        self._log.insert("end", msg + "\n")
        self._log.see("end")
        self._log.config(state="disabled")

    def _start(self):
        if self._running:
            return
        # Charge l'agent si pas encore fait
        if self._sim_act_fn is None and self._sim_combo.get() != "🎲 Random":
            self._on_sim_agent_select()

        self._running = True
        self._btn_start.config(state="disabled")
        self._btn_stop.config(state="normal")
        self._progress["value"] = 0
        n = self._n_var.get()
        label = self._sim_combo.get()
        self._log_append(f"[{time.strftime('%H:%M:%S')}] Lancement de {n} parties — {label}…")
        self._thread = threading.Thread(
            target=self._run_sim, args=(n,), daemon=True
        )
        self._thread.start()

    def _stop(self):
        self._running = False

    def _run_sim(self, n: int):
        env    = TicTacToe()
        wins_x = wins_o = draws = 0
        t0     = time.perf_counter()
        update_every = max(1, n // 200)
        act_fn = self._sim_act_fn  # snapshot thread-safe

        for i in range(n):
            if not self._running:
                break
            env.reset()
            while not env.is_game_over():
                if act_fn is not None:
                    try:
                        action = act_fn(env)
                    except Exception:
                        action = random.choice(env.available_actions())
                else:
                    action = random.choice(env.available_actions())
                env.step(action)

            if env._winner == 0:   wins_x += 1
            elif env._winner == 1: wins_o += 1
            else:                  draws  += 1

            if (i + 1) % update_every == 0:
                elapsed = time.perf_counter() - t0
                speed   = (i + 1) / elapsed if elapsed > 0 else 0
                self.after(0, self._update_ui,
                           i + 1, speed, wins_x, wins_o, draws, (i+1)/n*100)

        elapsed = time.perf_counter() - t0
        total   = wins_x + wins_o + draws
        speed   = total / elapsed if elapsed > 0 else 0
        self.after(0, self._finish_ui, total, speed, wins_x, wins_o, draws, elapsed)

    def _update_ui(self, games, speed, wins_x, wins_o, draws, pct):
        if not self._running:
            return
        total = wins_x + wins_o + draws or 1
        self._progress["value"] = pct
        self._prog_lbl.config(text=f"{games:,} / {self._n_var.get():,}")
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
        self._prog_lbl.config(text=f"{total:,} parties en {elapsed:.2f}s")
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

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG2, foreground=TXT,
                        padding=[16, 8], font=("Segoe UI", 11))
        style.map("TNotebook.Tab",
                  background=[("selected", BG3)],
                  foreground=[("selected", GOLD)])

        for name, color in [("W", X_COL), ("L", O_COL), ("D", GOLD), ("Sim", GOLD)]:
            style.configure(f"{name}.Horizontal.TProgressbar",
                            troughcolor=BG3, background=color,
                            thickness=8, borderwidth=0)

        style.configure("TScrollbar", background=BG3, troughcolor=BG2, borderwidth=0)

    def _build_notebook(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        game_tab = GameTab(nb)
        sim_tab  = SimulationTab(nb)

        nb.add(game_tab, text="  🎮  Jeu  ")
        nb.add(sim_tab,  text="  📊  Simulation  ")


if __name__ == "__main__":
    app = TicTacToeApp()
    app.mainloop()