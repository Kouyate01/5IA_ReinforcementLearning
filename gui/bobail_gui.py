"""
gui/bobail_gui.py
=================
Interface graphique tkinter pour l'environnement Bobail.

Modes :
  • Human vs Human   — 2 joueurs humains, alternance X/O
  • Agent vs Random  — agent RL joue X, Random joue O (auto-play)
  • Agent vs Human   — agent RL joue X, humain joue O
Onglet Simulation   — benchmark agent vs Random sur N parties

Règles :
  Phase 0 : déplacer le Bobail d'1 case dans 8 directions (case vide)
  Phase 1 : glisser un pion aussi loin que possible dans une direction
  Victoire : Bobail sur sa ligne de base OU Bobail complètement bloqué

Encodage des actions (identique à env.Bobail) :
  Phase 0 : action = direction (0-7)
  Phase 1 : action = 8 + (row*5+col)*8 + direction

Agents supportés : TQL, DQN, DDQN, DDQNER, DDQNPER, ExpertApprentice, PPO
"""

import sys
import os
import time
import random
import threading
import importlib
import pickle
import numpy as np
import torch
import tkinter as tk
from tkinter import ttk
from typing import Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.bobail import Bobail

# ─────────────────────────────────────────────────────────────────────────────
# Registre & chargement d'agents (même pattern que tictactoe_gui.py)
# ─────────────────────────────────────────────────────────────────────────────

AGENT_REGISTRY = {
    "dqn":              ("agents.dqn",     "DeepQLearning",                                      "rl"),
    "ddqn":             ("agents.ddqn",    "DoubleDeepQLearning",                                 "rl"),
    "ddqner":           ("agents.ddqner",  "DoubleDeepQLearningWithExperienceReplay",             "rl"),
    "ddqnper":          ("agents.ddqnper", "DoubleDeepQLearningWithPrioritizedExperienceReplay",  "rl"),
    "tql":              ("agents.tabular_q_learning", "TabularQLearning",                         "rl"),
    "reinforce":        ("agents.reinforce", "REINFORCE",                                         "rl"),
    "reinforce_critic": ("agents.reinforce_critic", "REINFORCEWithCritic",                        "rl"),
    "reinforce_mb":     ("agents.reinforce_mean_baseline", "REINFORCEMeanBaseline",               "rl"),
    "apprentice":       ("agents.expert_apprentice",  "ExpertApprenticeAgent",                    "act"),
    "ppo":              ("agents.ppo",     "PPOAgent",                                            "act"),
}


def _infer_sizes_from_checkpoint(model_path: str, interface: str):
    if interface != "act":
        return None, None
    try:
        sd = torch.load(model_path, map_location="cpu")
        if "fc.0.weight" in sd:
            state_size  = sd["fc.0.weight"].shape[1]
            last_w      = sorted(k for k in sd if k.endswith(".weight"))[-1]
            action_size = sd[last_w].shape[0]
        elif "shared.0.weight" in sd:
            state_size  = sd["shared.0.weight"].shape[1]
            action_size = sd["actor.0.weight"].shape[0]
        else:
            return None, None
        return int(state_size), int(action_size)
    except Exception as e:
        print(f"⚠️  Inférence échouée : {e}")
        return None, None


def load_agent_fn(agent_type: str, model_path: str, env) -> callable:
    """Retourne une callable act(env_ref) -> int, ou None."""
    if agent_type not in AGENT_REGISTRY:
        print(f"⚠️  Type inconnu : {agent_type}")
        return None

    module_name, class_name, interface = AGENT_REGISTRY[agent_type]
    try:
        cls = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as e:
        print(f"❌ Import {class_name} : {e}")
        return None

    try:
        if interface == "act":
            state_size, action_size = _infer_sizes_from_checkpoint(model_path, interface)
            if state_size is None:
                state_size, action_size = env.state_size, env.action_size
            agent = cls(state_size, action_size, model_path=model_path)
            return lambda env_ref: agent.act(env_ref, render=True)

        if agent_type == "tql":
            with open(model_path, "rb") as f:
                data = pickle.load(f)
            agent = cls(
                alpha=data["alpha"], gamma=data["gamma"],
                epsilon=0.0, epsilon_min=0.0, epsilon_decay=1.0,
            )
            agent._q_table = data["q_table"]
        else:
            data = torch.load(model_path, map_location="cpu")
            data.update(epsilon=0.0, epsilon_min=0.0, epsilon_decay=1.0)
            agent = cls.from_config({})
            agent.load_state_dict(data, env.state_size, env.action_size)

        return lambda env_ref: agent.select_action(env_ref, greedy=True)

    except Exception as e:
        print(f"❌ Chargement {model_path} : {e}")
        return None


def scan_saved_models(base_dir: str, env) -> dict:
    """Scanne base_dir → {label: loader_fn | None}. None = Random."""
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
            label = f"{folder_name} › {fname}"
            def make_fn(t=agent_type, p=model_path):
                return load_agent_fn(t, p, env)
            agents[label] = make_fn
    return agents


# ── Palette ───────────────────────────────────────────────────────────────────
BG       = "#0e0f14"
BG2      = "#16171f"
BG3      = "#1e1f2b"
GOLD     = "#e8c547"
GOLD_DIM = "#a0893a"
X_COL    = "#e05c5c"
O_COL    = "#5ca8e0"
B_COL    = "#b08edb"
TXT      = "#d4d4d8"
TXT_DIM  = "#6b7280"
GRID_COL = "#2a2b3d"

CELL     = 80
PADDING  = 18
CANVAS_W = CELL * 5 + PADDING * 2
CANVAS_H = CELL * 5 + PADDING * 2
RADIUS   = 26
B_RADIUS = 20

DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1),
              (1, 0),  (1, -1), (0, -1), (-1, -1)]
DIR_NAMES  = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def cell_xy(row, col):
    return PADDING + col * CELL + CELL // 2, PADDING + row * CELL + CELL // 2

def xy_to_cell(x, y):
    col = (x - PADDING) // CELL
    row = (y - PADDING) // CELL
    if 0 <= row < 5 and 0 <= col < 5:
        return row, col
    return None

def slide_destination(board, from_r, from_c, direction):
    """Réplique de Bobail._slide_destination pour le GUI (sans modifier l'env)."""
    dr, dc = DIRECTIONS[direction]
    cur_r, cur_c = from_r, from_c
    while True:
        nr, nc = cur_r + dr, cur_c + dc
        if not (0 <= nr < 5 and 0 <= nc < 5):
            break
        if board[nr][nc] != 0:
            break
        cur_r, cur_c = nr, nc
    return cur_r, cur_c

def encode_phase1_action(row, col, direction):
    return 8 + (row * 5 + col) * 8 + direction

def decode_phase1_action(action):
    encoded   = action - 8
    case_idx  = encoded // 8
    direction = encoded  % 8
    return case_idx // 5, case_idx % 5, direction


# ─────────────────────────────────────────────────────────────────────────────
# Canvas
# ─────────────────────────────────────────────────────────────────────────────

class BobailCanvas(tk.Canvas):
    def __init__(self, parent, click_callback=None, **kwargs):
        super().__init__(parent, width=CANVAS_W, height=CANVAS_H,
                         bg=BG2, highlightthickness=0, **kwargs)
        self._click_cb = click_callback
        self.bind("<Button-1>", self._on_click)
        self._draw_grid()

    def _draw_grid(self):
        self.delete("grid")
        for r in range(5):
            for c in range(5):
                x0, y0 = PADDING + c * CELL, PADDING + r * CELL
                fill = BG2 if (r + c) % 2 == 0 else "#13141c"
                self.create_rectangle(x0, y0, x0+CELL, y0+CELL,
                                      fill=fill, outline=GRID_COL, width=1, tags="grid")
        # Zones victoire
        for c in range(5):
            x0 = PADDING + c * CELL
            self.create_rectangle(x0, PADDING, x0+CELL, PADDING+CELL,
                                  fill="#0e1a28", outline=GRID_COL, width=1, tags="grid")
            self.create_rectangle(x0, PADDING+4*CELL, x0+CELL, PADDING+5*CELL,
                                  fill="#1f1010", outline=GRID_COL, width=1, tags="grid")
        for i in range(5):
            self.create_text(PADDING+i*CELL+CELL//2, PADDING//2,
                             text=str(i), fill=TXT_DIM, font=("Consolas", 8), tags="grid")
            self.create_text(PADDING//2-2, PADDING+i*CELL+CELL//2,
                             text=str(i), fill=TXT_DIM, font=("Consolas", 8), tags="grid")

    def _on_click(self, event):
        if self._click_cb:
            cell = xy_to_cell(event.x, event.y)
            if cell is not None:
                self._click_cb(cell)

    def render(self, board, highlighted=None, selected=None):
        self.delete("piece", "highlight", "select", "winmark")
        if highlighted:
            for (r, c) in highlighted:
                x0, y0 = PADDING+c*CELL+4, PADDING+r*CELL+4
                self.create_rectangle(x0, y0, x0+CELL-8, y0+CELL-8,
                                      fill="", outline=GOLD, width=2,
                                      dash=(4, 3), tags="highlight")
        if selected:
            sr, sc = selected
            x0, y0 = PADDING+sc*CELL+2, PADDING+sr*CELL+2
            self.create_rectangle(x0, y0, x0+CELL-4, y0+CELL-4,
                                  fill=BG3, outline=GOLD, width=3, tags="select")
        for r in range(5):
            for c in range(5):
                val = board[r][c]
                if val == 0:
                    continue
                cx, cy = cell_xy(r, c)
                if val == 3:
                    self._draw_bobail(cx, cy)
                elif val == 1:
                    self._draw_pion(cx, cy, X_COL, "X")
                elif val == 2:
                    self._draw_pion(cx, cy, O_COL, "O")

    def _draw_pion(self, cx, cy, color, symbol):
        r = RADIUS
        self.create_oval(cx-r, cy-r, cx+r, cy+r,
                         fill=BG3, outline=color, width=3, tags="piece")
        self.create_text(cx, cy, text=symbol, fill=color,
                         font=("Segoe UI", 14, "bold"), tags="piece")

    def _draw_bobail(self, cx, cy):
        r = B_RADIUS
        self.create_oval(cx-r-6, cy-r-6, cx+r+6, cy+r+6,
                         fill="", outline=B_COL, width=1, dash=(3, 3), tags="piece")
        self.create_oval(cx-r, cy-r, cx+r, cy+r,
                         fill="#251e38", outline=B_COL, width=3, tags="piece")
        self.create_text(cx, cy, text="B", fill=B_COL,
                         font=("Segoe UI", 13, "bold"), tags="piece")

    def mark_winner(self, player):
        msg = "🏆 Joueur X gagne !" if player == 0 else "🏆 Joueur O gagne !"
        col = X_COL if player == 0 else O_COL
        cx, cy = CANVAS_W//2, CANVAS_H//2
        self.create_rectangle(cx-155, cy-28, cx+155, cy+28,
                              fill=BG, outline=col, width=3, tags="winmark")
        self.create_text(cx, cy, text=msg, fill=col,
                         font=("Segoe UI", 15, "bold"), tags="winmark")

    def clear(self):
        self.delete("piece", "highlight", "select", "winmark")


# ─────────────────────────────────────────────────────────────────────────────
# Widgets annexes
# ─────────────────────────────────────────────────────────────────────────────

class StateVectorPanel(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG3, padx=8, pady=6)
        tk.Label(self, text="Vecteur d'état (77,)", bg=BG3, fg=GOLD,
                 font=("Consolas", 9, "bold")).pack(anchor="w")
        self._lbl = tk.Label(self, text="—", bg=BG3, fg=TXT,
                             font=("Consolas", 8), justify="left", wraplength=380)
        self._lbl.pack(anchor="w", pady=(2, 4))
        tk.Label(self, text="Dernière action", bg=BG3, fg=GOLD_DIM,
                 font=("Consolas", 9, "bold")).pack(anchor="w")
        self._lbl_a = tk.Label(self, text="—", bg=BG3, fg=TXT_DIM,
                               font=("Consolas", 8), justify="left", wraplength=380)
        self._lbl_a.pack(anchor="w")

    def update(self, state, action=None):
        lines = []
        for name, offset in [("J0", 0), ("J1", 25), ("B ", 50)]:
            chunk = " ".join(f"{int(v)}" for v in state[offset:offset+25])
            lines.append(f"{name} [{chunk}]")
        lines.append(f"cur={int(state[75])} phase={int(state[76])}")
        self._lbl.config(text="\n".join(lines))
        if action is not None:
            if action < 8:
                self._lbl_a.config(text=f"Phase 0 — direction {action} ({DIR_NAMES[action]})")
            else:
                r, c, d = decode_phase1_action(action)
                self._lbl_a.config(text=f"Phase 1 — pion ({r},{c}) → {DIR_NAMES[d]}  [a={action}]")
        else:
            self._lbl_a.config(text="—")

    def clear(self):
        self._lbl.config(text="—")
        self._lbl_a.config(text="—")


class SessionStatsBar(tk.Frame):
    def __init__(self, parent, **kwargs):
        # Ne pas passer **kwargs à tk.Frame pour éviter la corruption du widget
        super().__init__(parent, bg=BG)
        self.wins = self.losses = self.draws = 0
        lbl = dict(bg=BG, fg=TXT, font=("Segoe UI", 9))
        big = dict(bg=BG, font=("Segoe UI", 12, "bold"))
        tk.Label(self, text="SESSION", bg=BG, fg=GOLD,
                 font=("Segoe UI", 8, "bold")).grid(row=0, column=0, columnspan=9, pady=(0,4))
        tk.Label(self, text="✔ Victoires", **lbl).grid(row=1, column=0, sticky="w")
        self._lbl_w = tk.Label(self, text="0", fg=X_COL, **big)
        self._lbl_w.grid(row=1, column=1, padx=(4,12))
        self._bar_w = ttk.Progressbar(self, length=80, maximum=100, style="W.Horizontal.TProgressbar")
        self._bar_w.grid(row=1, column=2, padx=(0,14))
        tk.Label(self, text="✘ Défaites", **lbl).grid(row=1, column=3, sticky="w")
        self._lbl_l = tk.Label(self, text="0", fg=O_COL, **big)
        self._lbl_l.grid(row=1, column=4, padx=(4,12))
        self._bar_l = ttk.Progressbar(self, length=80, maximum=100, style="L.Horizontal.TProgressbar")
        self._bar_l.grid(row=1, column=5, padx=(0,14))
        tk.Label(self, text="= Nuls", **lbl).grid(row=1, column=6, sticky="w")
        self._lbl_d = tk.Label(self, text="0", fg=GOLD, **big)
        self._lbl_d.grid(row=1, column=7, padx=(4,12))
        self._bar_d = ttk.Progressbar(self, length=80, maximum=100, style="D.Horizontal.TProgressbar")
        self._bar_d.grid(row=1, column=8)

    def record(self, result):
        if result == "win":    self.wins   += 1
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
# Onglet Jeu
# ─────────────────────────────────────────────────────────────────────────────

class GameTab(tk.Frame):

    MODES = [
        "Human vs Human",
        "Agent vs Random",
        "Agent vs Human",
    ]

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG)
        self._env       = Bobail()
        self._mode      = tk.StringVar(value=self.MODES[0])
        self._status    = tk.StringVar(value="")
        self._auto_job  = None
        self._paused    = False
        self._speed_ms  = 900   # délai entre steps agent (ms)
        self._game_over = False
        self._last_action: Optional[int] = None

        # Sélection pion (phase 1 humain)
        self._sel_pion: Optional[tuple]  = None
        self._pion_targets: dict         = {}   # {(dest_r, dest_c): action_int}

        # Agent chargé
        base = os.path.join(ROOT, "modeles_gui", "bobail")
        self._agent_catalog  = scan_saved_models(base, self._env)
        self._current_act_fn: Optional[callable] = None

        self._build_ui()
        self._populate_agent_combo()
        self._new_game()

    # ─────────────────────────────────────────────────────────────────────────
    # UI
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        header = tk.Frame(self, bg=BG)
        header.pack(fill="x", padx=20, pady=(14, 0))
        tk.Label(header, text="Bobail", bg=BG, fg=GOLD,
                 font=("Segoe UI", 20, "bold")).pack(side="left")

        mode_f = tk.Frame(header, bg=BG)
        mode_f.pack(side="right")
        for m in self.MODES:
            tk.Radiobutton(mode_f, text=m, variable=self._mode, value=m,
                           bg=BG, fg=TXT, selectcolor=BG2,
                           activebackground=BG, activeforeground=GOLD,
                           font=("Segoe UI", 10),
                           command=self._on_mode_change).pack(side="left", padx=6)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=20, pady=10)

        # ── Gauche : plateau ──
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", anchor="n")

        self._canvas = BobailCanvas(left, click_callback=self._on_cell_click)
        self._canvas.pack()

        leg = tk.Frame(left, bg=BG)
        leg.pack(pady=(4, 0))
        for sym, col, lbl in [("X", X_COL, "J0 → gagne ligne 4"),
                               ("O", O_COL, "J1 → gagne ligne 0"),
                               ("B", B_COL, "Bobail")]:
            tk.Label(leg, text=f"{sym} ", bg=BG, fg=col,
                     font=("Segoe UI", 10, "bold")).pack(side="left")
            tk.Label(leg, text=f"{lbl}   ", bg=BG, fg=TXT_DIM,
                     font=("Segoe UI", 9)).pack(side="left")

        self._status_lbl = tk.Label(left, textvariable=self._status,
                                    bg=BG, fg=GOLD, font=("Segoe UI", 11, "bold"),
                                    height=2, width=44, wraplength=400, justify="center")
        self._status_lbl.pack(pady=6)

        btn_row = tk.Frame(left, bg=BG)
        btn_row.pack(pady=(0, 4))
        tk.Button(btn_row, text="⟳  Nouvelle Partie",
                  bg=BG3, fg=GOLD, activebackground=GOLD, activeforeground=BG,
                  font=("Segoe UI", 10, "bold"), relief="flat", padx=16, pady=6,
                  cursor="hand2", command=self._new_game).pack(side="left", padx=(0, 8))
        self._btn_pause = tk.Button(btn_row, text="⏸  Pause",
                  bg=BG3, fg=TXT, activebackground=BG2,
                  font=("Segoe UI", 10, "bold"), relief="flat", padx=16, pady=6,
                  cursor="hand2", state="disabled", command=self._toggle_pause)
        self._btn_pause.pack(side="left")

        # ── Droite : contrôles ──
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(20, 0), anchor="n")

        # Phase indicator
        self._phase_lbl = tk.Label(right, text="", bg=BG3, fg=B_COL,
                                   font=("Segoe UI", 10, "bold"), pady=6, padx=10)
        self._phase_lbl.pack(fill="x", pady=(0, 8))

        # Boutons directionnels (Phase 0)
        dir_outer = tk.LabelFrame(right, text=" Phase 0 — Direction du Bobail ",
                                  bg=BG, fg=GOLD_DIM, font=("Segoe UI", 9),
                                  labelanchor="n")
        dir_outer.pack(fill="x", pady=(0, 8))

        self._dir_btns: dict = {}
        layout = [(0,0,7,"NO"),(0,1,0,"N"),(0,2,1,"NE"),
                  (1,0,6,"O"),             (1,2,2,"E"),
                  (2,0,5,"SO"),(2,1,4,"S"),(2,2,3,"SE")]
        for gr, gc, d, name in layout:
            btn = tk.Button(dir_outer, text=name, width=4,
                            bg=BG3, fg=TXT, relief="flat",
                            font=("Segoe UI", 9), state="disabled",
                            command=lambda d=d: self._human_bobail_move(d))
            btn.grid(row=gr, column=gc, padx=2, pady=2)
            self._dir_btns[d] = btn
        tk.Label(dir_outer, text="B", bg=BG3, fg=B_COL,
                 font=("Segoe UI", 11, "bold"), width=4).grid(row=1, column=1)

        # Sélection d'agent
        agent_f = tk.LabelFrame(right, text=" Agent (Bobail) ",
                                bg=BG, fg=GOLD, font=("Segoe UI", 9), padx=10, pady=8)
        agent_f.pack(fill="x", pady=(0, 8))

        self._agent_combo = ttk.Combobox(agent_f, state="readonly")
        self._agent_combo.pack(fill="x")
        self._agent_combo.bind("<<ComboboxSelected>>", self._on_agent_select)

        self._agent_status_lbl = tk.Label(agent_f, text="Aucun agent chargé",
                                          bg=BG, fg=TXT_DIM, font=("Segoe UI", 9))
        self._agent_status_lbl.pack(anchor="w", pady=(4, 0))

        # Vecteur état
        self._vec_panel = StateVectorPanel(right)
        self._vec_panel.pack(fill="x", pady=(0, 10))

        # Stats
        tk.Label(right, text="Statistiques de session", bg=BG, fg=GOLD,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self._stats = SessionStatsBar(right)
        self._stats.pack(anchor="w")
        tk.Button(right, text="Réinitialiser", bg=BG3, fg=TXT_DIM,
                  font=("Segoe UI", 9), relief="flat", cursor="hand2",
                  command=self._stats.reset).pack(anchor="w", pady=(6, 0))

    def _populate_agent_combo(self):
        labels = list(self._agent_catalog.keys())
        self._agent_combo["values"] = labels
        if labels:
            self._agent_combo.set(labels[0])
            self._on_agent_select()

    # ─────────────────────────────────────────────────────────────────────────
    # Chargement agent
    # ─────────────────────────────────────────────────────────────────────────

    def _on_agent_select(self, event=None):
        label  = self._agent_combo.get()
        loader = self._agent_catalog.get(label)
        if loader is None:
            self._current_act_fn = None
            self._agent_status_lbl.config(text="Mode : Random", fg=TXT_DIM)
            return
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
    # Nouvelle partie
    # ─────────────────────────────────────────────────────────────────────────

    def _on_mode_change(self):
        self._cancel_auto()
        self._new_game()

    def _new_game(self):
        self._cancel_auto()
        self._env.reset()
        self._game_over    = False
        self._last_action  = None
        self._sel_pion     = None
        self._pion_targets = {}
        self._canvas.clear()
        self._canvas._draw_grid()
        self._vec_panel.clear()

        mode = self._mode.get()
        if mode in ("Agent vs Random", "Agent vs Human") or \
           (mode == "Human vs Random" and self._env._current_player == 1):
            # Si l'agent/random commence (impossible ici mais par sécurité)
            pass

        self._paused = False
        self._btn_pause.config(state="disabled", text="⏸  Pause")

        if mode == "Agent vs Random":
            self._status.set("Agent (X) réfléchit…")
            self._phase_lbl.config(text="")
            self._set_dir_buttons([])
            self._canvas.render(self._env._board)
            self._btn_pause.config(state="normal")
            self._schedule_agent_step()
        elif mode == "Agent vs Human":
            # L'agent (J0) commence toujours
            self._canvas.render(self._env._board)
            self._schedule_agent_step()
        else:
            self._refresh_display()

    def _cancel_auto(self):
        if self._auto_job:
            self.after_cancel(self._auto_job)
            self._auto_job = None

    def _toggle_pause(self):
        mode = self._mode.get()
        if mode not in ("Agent vs Random",):
            return
        self._paused = not self._paused
        if self._paused:
            self._cancel_auto()
            self._btn_pause.config(text="▶  Reprendre", fg=GOLD)
            self._status.set("⏸ En pause — cliquez Reprendre pour continuer")
        else:
            self._btn_pause.config(text="⏸  Pause", fg=TXT)
            self._schedule_agent_step()

    # ─────────────────────────────────────────────────────────────────────────
    # Affichage
    # ─────────────────────────────────────────────────────────────────────────

    def _refresh_display(self):
        env    = self._env
        player = env._current_player
        phase  = env._phase
        psym   = "X" if player == 0 else "O"
        pcol   = X_COL if player == 0 else O_COL
        phase_str = "Phase 1/2 — Déplacer le Bobail" if phase == 0 \
                    else "Phase 2/2 — Glisser un pion"
        self._phase_lbl.config(text=f"Tour de {psym}  ·  {phase_str}", fg=pcol)

        is_human = self._is_human_turn()
        legal    = env.available_actions()

        if phase == 0 and is_human:
            self._set_dir_buttons(legal)
            self._status.set(f"[{psym}] Choisissez une direction pour le Bobail")
        elif phase == 1 and is_human:
            self._set_dir_buttons([])
            if self._sel_pion:
                self._status.set(f"[{psym}] Cliquez sur la case de destination")
            else:
                self._status.set(f"[{psym}] Cliquez sur un de vos pions à glisser")
        else:
            self._set_dir_buttons([])

        highlights = set(self._pion_targets.keys()) if self._sel_pion else set()
        self._canvas.render(env._board, highlights, selected=self._sel_pion)
        self._vec_panel.update(env.get_state(), self._last_action)

    def _set_dir_buttons(self, legal_dirs):
        for d, btn in self._dir_btns.items():
            if d in legal_dirs:
                btn.config(state="normal", fg=GOLD)
            else:
                btn.config(state="disabled", fg=TXT_DIM)

    def _is_human_turn(self) -> bool:
        """Retourne True si c'est le tour d'un humain dans le mode courant."""
        mode   = self._mode.get()
        player = self._env._current_player
        if mode == "Human vs Human":
            return True
        if mode == "Agent vs Human":
            return player == 1    # humain = O = joueur 1
        return False              # Agent vs Random : jamais humain

    # ─────────────────────────────────────────────────────────────────────────
    # Destinations de glissement (pour affichage GUI)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_pion_targets(self, pion_rc: tuple) -> dict:
        env    = self._env
        pr, pc = pion_rc
        legal  = env.available_actions()
        targets = {}
        for d in range(8):
            action = encode_phase1_action(pr, pc, d)
            if action in legal:
                dest = slide_destination(env._board, pr, pc, d)
                if dest != (pr, pc):
                    targets[dest] = action
        return targets

    # ─────────────────────────────────────────────────────────────────────────
    # Interactions humain — Phase 0 (boutons directionnels)
    # ─────────────────────────────────────────────────────────────────────────

    def _human_bobail_move(self, direction: int):
        env = self._env
        if self._game_over or env._phase != 0 or not self._is_human_turn():
            return
        if direction not in env.available_actions():
            return
        self._execute_action(direction)

    # ─────────────────────────────────────────────────────────────────────────
    # Interactions humain — Phase 1 (clic pion + destination)
    # ─────────────────────────────────────────────────────────────────────────

    def _on_cell_click(self, cell: tuple):
        if self._game_over:
            return
        mode = self._mode.get()
        if mode == "Agent vs Random":
            return
        if not self._is_human_turn():
            return
        env = self._env
        if env._phase != 1:
            return

        r, c    = cell
        val     = env._board[r][c]
        player  = env._current_player
        own_val = player + 1

        # Clic sur une destination déjà calculée
        if self._sel_pion and cell in self._pion_targets:
            action = self._pion_targets[cell]
            self._sel_pion     = None
            self._pion_targets = {}
            self._execute_action(action)
            return

        # Clic sur un de ses pions → sélection
        if val == own_val:
            self._sel_pion     = cell
            self._pion_targets = self._compute_pion_targets(cell)
            self._refresh_display()
            return

        # Clic ailleurs → désélection
        self._sel_pion     = None
        self._pion_targets = {}
        self._refresh_display()

    # ─────────────────────────────────────────────────────────────────────────
    # Exécution d'une action (humain OU agent)
    # ─────────────────────────────────────────────────────────────────────────

    def _execute_action(self, action: int):
        """Joue `action` dans l'env puis décide qui joue ensuite."""
        self._last_action  = action
        self._sel_pion     = None
        self._pion_targets = {}
        _, _, done = self._env.step(action)

        if done:
            self._finish_game()
            return

        # Décision post-step
        mode = self._mode.get()
        if self._is_human_turn():
            # Tour humain : afficher et attendre un clic/bouton
            self._refresh_display()
        elif mode in ("Agent vs Random", "Agent vs Human"):
            # Tour agent
            self._refresh_display()
            self._schedule_agent_step()
        else:
            self._refresh_display()

    # ─────────────────────────────────────────────────────────────────────────
    # Step random (une phase)
    # ─────────────────────────────────────────────────────────────────────────

    def _random_step(self):
        env = self._env
        if self._game_over or env.is_game_over():
            return
        actions = env.available_actions()
        if not actions:
            return
        action = random.choice(actions)
        self._execute_action(action)

    # ─────────────────────────────────────────────────────────────────────────
    # Step agent
    # ─────────────────────────────────────────────────────────────────────────

    def _schedule_agent_step(self):
        if not self._paused:
            self._auto_job = self.after(self._speed_ms, self._agent_step)

    def _agent_step(self):
        self._auto_job = None
        if self._game_over:
            return
        env = self._env
        actions = env.available_actions()
        if not actions:
            return

        # Choisir l'action
        if self._current_act_fn is not None:
            try:
                action = self._current_act_fn(env)
            except Exception as e:
                print(f"⚠️  Erreur agent : {e} — fallback Random")
                action = random.choice(actions)
        else:
            action = random.choice(actions)

        # Vérifier que l'action est légale (sécurité)
        if action not in actions:
            action = random.choice(actions)

        self._execute_action(action)

    # ─────────────────────────────────────────────────────────────────────────
    # Fin de partie
    # ─────────────────────────────────────────────────────────────────────────

    def _finish_game(self):
        self._game_over = True
        env = self._env
        self._canvas.render(env._board)
        self._set_dir_buttons([])
        self._phase_lbl.config(text="Partie terminée", fg=GOLD)

        if env._winner == 0:
            self._canvas.mark_winner(0)
            self._status.set("🏆 Joueur X (0) gagne !")
            self._stats.record("win")
        elif env._winner == 1:
            self._canvas.mark_winner(1)
            self._status.set("🏆 Joueur O (1) gagne !")
            self._stats.record("loss")
        else:
            self._status.set("Match nul !")
            self._stats.record("draw")


# ─────────────────────────────────────────────────────────────────────────────
# Onglet Simulation
# ─────────────────────────────────────────────────────────────────────────────

class SimulationTab(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG)
        self._running = False
        self._thread  = None

        self._env_ref        = Bobail()
        base                 = os.path.join(ROOT, "saved_models", "bobail")
        self._agent_catalog  = scan_saved_models(base, self._env_ref)
        self._sim_act_fn     = None

        self._build_ui()

    def _build_ui(self):
        tk.Label(self, text="Benchmark — Agent vs Random", bg=BG, fg=GOLD,
                 font=("Segoe UI", 18, "bold")).pack(anchor="w", padx=20, pady=(16, 6))

        cfg = tk.Frame(self, bg=BG2, padx=16, pady=12)
        cfg.pack(fill="x", padx=20, pady=(0, 10))

        tk.Label(cfg, text="Parties :", bg=BG2, fg=TXT,
                 font=("Segoe UI", 11)).grid(row=0, column=0, sticky="w")
        self._n_var = tk.IntVar(value=5000)
        tk.Spinbox(cfg, from_=100, to=500000, increment=500,
                   textvariable=self._n_var, width=10,
                   bg=BG3, fg=TXT, buttonbackground=BG3,
                   font=("Segoe UI", 11), relief="flat").grid(row=0, column=1, padx=10)

        # Sélection agent
        tk.Label(cfg, text="Agent :", bg=BG2, fg=TXT,
                 font=("Segoe UI", 11)).grid(row=0, column=2, sticky="w", padx=(16, 0))
        self._sim_combo = ttk.Combobox(cfg, values=list(self._agent_catalog.keys()),
                                       state="readonly", width=32)
        self._sim_combo.grid(row=0, column=3, padx=8)
        labels = list(self._agent_catalog.keys())
        if labels:
            self._sim_combo.set(labels[0])
        self._sim_combo.bind("<<ComboboxSelected>>", self._on_sim_agent_select)

        self._btn_start = tk.Button(cfg, text="▶  Lancer",
                                    bg=GOLD, fg=BG, font=("Segoe UI", 11, "bold"),
                                    relief="flat", padx=16, pady=6, cursor="hand2",
                                    command=self._start)
        self._btn_start.grid(row=0, column=4, padx=10)

        self._btn_stop = tk.Button(cfg, text="■  Arrêter",
                                   bg=BG3, fg=TXT, font=("Segoe UI", 11),
                                   relief="flat", padx=16, pady=6, cursor="hand2",
                                   state="disabled", command=self._stop)
        self._btn_stop.grid(row=0, column=5)

        self._sim_status_lbl = tk.Label(cfg, text="", bg=BG2, fg=TXT_DIM,
                                         font=("Segoe UI", 9))
        self._sim_status_lbl.grid(row=1, column=2, columnspan=4, sticky="w", pady=(6, 0))

        self._progress = ttk.Progressbar(self, length=700, maximum=100,
                                         style="Sim.Horizontal.TProgressbar")
        self._progress.pack(padx=20, pady=(0, 4))
        self._prog_lbl = tk.Label(self, text="", bg=BG, fg=TXT_DIM, font=("Segoe UI", 9))
        self._prog_lbl.pack()

        res = tk.Frame(self, bg=BG2, padx=20, pady=16)
        res.pack(fill="x", padx=20, pady=10)

        def stat_col(parent, label, color, col):
            f = tk.Frame(parent, bg=BG2)
            f.grid(row=0, column=col, padx=20)
            tk.Label(f, text=label, bg=BG2, fg=TXT_DIM, font=("Segoe UI", 9)).pack()
            lbl = tk.Label(f, text="—", bg=BG2, fg=color, font=("Segoe UI", 22, "bold"))
            lbl.pack()
            return lbl

        self._lbl_games = stat_col(res, "Parties",     TXT,   0)
        self._lbl_speed = stat_col(res, "Parties/sec", GOLD,  1)
        self._lbl_j0    = stat_col(res, "% X gagne",  X_COL, 2)
        self._lbl_j1    = stat_col(res, "% O gagne",  O_COL, 3)
        self._lbl_draw  = stat_col(res, "% Nuls",     GOLD,  4)

        log_f = tk.Frame(self, bg=BG2)
        log_f.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        self._log = tk.Text(log_f, bg=BG2, fg=TXT_DIM, font=("Consolas", 9),
                            relief="flat", height=8, state="disabled")
        sc = ttk.Scrollbar(log_f, command=self._log.yview)
        self._log.configure(yscrollcommand=sc.set)
        self._log.pack(side="left", fill="both", expand=True)
        sc.pack(side="right", fill="y")

    def _on_sim_agent_select(self, event=None):
        label  = self._sim_combo.get()
        loader = self._agent_catalog.get(label)
        if loader is None:
            self._sim_act_fn = None
            self._sim_status_lbl.config(text="Mode : Random vs Random", fg=TXT_DIM)
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

    def _log_write(self, msg):
        self._log.config(state="normal")
        self._log.insert("end", msg + "\n")
        self._log.see("end")
        self._log.config(state="disabled")

    def _start(self):
        if self._running:
            return
        if self._sim_act_fn is None and self._sim_combo.get() != "🎲 Random":
            self._on_sim_agent_select()
        self._running = True
        self._btn_start.config(state="disabled")
        self._btn_stop.config(state="normal")
        self._progress["value"] = 0
        n     = self._n_var.get()
        label = self._sim_combo.get()
        self._log_write(f"[{time.strftime('%H:%M:%S')}] {n} parties — {label}…")
        self._thread = threading.Thread(target=self._run, args=(n,), daemon=True)
        self._thread.start()

    def _stop(self):
        self._running = False

    def _run(self, n):
        """
        L'agent (act_fn) joue en tant que J0 (X).
        À chaque phase, si c'est le tour de J0 → agent, sinon → random.
        """
        env    = Bobail()
        act_fn = self._sim_act_fn   # snapshot thread-safe
        j0 = j1 = draws = 0
        t0    = time.perf_counter()
        every = max(1, n // 200)

        for i in range(n):
            if not self._running:
                break
            env.reset()
            while not env.is_game_over():
                actions = env.available_actions()
                if not actions:
                    break
                # J0 = agent, J1 = random
                if env._current_player == 0 and act_fn is not None:
                    try:
                        action = act_fn(env)
                        if action not in actions:
                            action = random.choice(actions)
                    except Exception:
                        action = random.choice(actions)
                else:
                    action = random.choice(actions)
                env.step(action)

            if env._winner == 0:    j0    += 1
            elif env._winner == 1:  j1    += 1
            else:                   draws += 1

            if (i + 1) % every == 0:
                elapsed = time.perf_counter() - t0
                speed   = (i + 1) / elapsed if elapsed > 0 else 0
                self.after(0, self._update_ui,
                           i+1, speed, j0, j1, draws, (i+1)/n*100)

        elapsed = time.perf_counter() - t0
        total   = j0 + j1 + draws
        speed   = total / elapsed if elapsed > 0 else 0
        self.after(0, self._done_ui, total, speed, j0, j1, draws, elapsed)

    def _update_ui(self, games, speed, j0, j1, draws, pct):
        if not self._running:
            return
        total = j0 + j1 + draws or 1
        self._progress["value"] = pct
        self._prog_lbl.config(text=f"{games:,} / {self._n_var.get():,}")
        self._lbl_games.config(text=f"{games:,}")
        self._lbl_speed.config(text=f"{speed:,.0f}")
        self._lbl_j0.config(text=f"{j0/total*100:.1f}%")
        self._lbl_j1.config(text=f"{j1/total*100:.1f}%")
        self._lbl_draw.config(text=f"{draws/total*100:.1f}%")

    def _done_ui(self, total, speed, j0, j1, draws, elapsed):
        self._running = False
        self._btn_start.config(state="normal")
        self._btn_stop.config(state="disabled")
        self._progress["value"] = 100
        self._prog_lbl.config(text=f"{total:,} parties en {elapsed:.2f}s")
        self._update_ui(total, speed, j0, j1, draws, 100)
        self._log_write(
            f"[{time.strftime('%H:%M:%S')}] Terminé — {total:,} parties | "
            f"{speed:,.0f} p/s | X={j0/total*100:.1f}% | "
            f"O={j1/total*100:.1f}% | ={draws/total*100:.1f}%"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Application
# ─────────────────────────────────────────────────────────────────────────────

class BobailApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bobail — Reinforcement Learning")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(980, 700)
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
        game_tab = GameTab(self)
        sim_tab  = SimulationTab(self)
        nb.add(game_tab, text="  🎮  Jeu  ")
        nb.add(sim_tab,  text="  📊  Simulation  ")


if __name__ == "__main__":
    app = BobailApp()
    app.mainloop()