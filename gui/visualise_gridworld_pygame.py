"""
gui/visualise_gridworld_pygame.py
==================================
Visualisation tkinter de GridWorld avec agent Dyna-Q.
(Fichier nommé *_pygame par convention du projet mais utilise tkinter,
 pygame n'étant pas compatible avec Python 3.14+)

GridWorld 5×5 :
  - Départ : case (0,0) haut-gauche
  - Objectif : case (4,4) bas-droite
  - Actions : Up(0) Down(1) Left(2) Right(3)
  - Reward : +1.0 à l'arrivée, -0.01 à chaque pas
"""

import sys
import os
import time
import random
import threading
from collections import defaultdict
import numpy as np
import tkinter as tk
from tkinter import ttk

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.grid_world import GridWorld

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = "#0e0f14"
BG2      = "#16171f"
BG3      = "#1e1f2b"
GOLD     = "#e8c547"
GOLD_DIM = "#a0893a"
TXT      = "#d4d4d8"
TXT_DIM  = "#6b7280"
AGENT_C  = "#5ca8e0"
GOAL_C   = "#4caf76"
CELL_C   = "#1e1f2b"
CELL_C2  = "#191a25"
ARROW_C  = "#e8c547"
OPT_C    = "#e05c5c"
PATH_C   = "#b08edb"

CELL  = 90
ROWS  = 5
COLS  = 5
PAD   = 24
CW    = COLS * CELL + 2 * PAD
CH    = ROWS * CELL + 2 * PAD

ACTION_SYMBOLS = {0: "↑", 1: "↓", 2: "←", 3: "→"}

# ── Dyna-Q ─────────────────────────────────────────────────────────────────────

def dyna_q(env, episodes=400, alpha=0.1, gamma=0.95, epsilon=0.15, n_planning=15):
    Q = defaultdict(lambda: defaultdict(float))
    Model = {}
    visited = []
    rewards = []

    for _ in range(episodes):
        state = tuple(env.reset())
        done = False
        total = 0
        while not done:
            actions = env.available_actions()
            if random.random() < epsilon:
                a = random.choice(actions)
            else:
                qs = [Q[state][a] for a in actions]
                a = actions[int(np.argmax(qs))]
            ns_arr, r, done = env.step(a)
            ns = tuple(ns_arr)
            total += r
            best_next = max([Q[ns][na] for na in env.available_actions()] or [0.0])
            Q[state][a] += alpha * (r + gamma * best_next - Q[state][a])
            Model[(state, a)] = (r, ns)
            if (state, a) not in visited:
                visited.append((state, a))
            for _ in range(n_planning):
                s_p, a_p = random.choice(visited)
                r_p, ns_p = Model[(s_p, a_p)]
                env_temp = GridWorld(ROWS, COLS)
                best_p = max([Q[ns_p][na] for na in env_temp.available_actions()] or [0.0])
                Q[s_p][a_p] += alpha * (r_p + gamma * best_p - Q[s_p][a_p])
            state = ns
        rewards.append(total)

    # Politique greedy
    env.reset()
    policy = {}
    for r in range(ROWS):
        for c in range(COLS):
            s = tuple(np.zeros(ROWS * COLS)); lst = list(s); lst[r * COLS + c] = 1.0; s = tuple(lst)
            actions = GridWorld(ROWS, COLS).available_actions()
            qs = [(a, Q[s][a]) for a in actions]
            policy[(r, c)] = max(qs, key=lambda x: x[1])[0] if qs else None
    return Q, policy, rewards


def value_iteration(env, gamma=0.95, theta=1e-6):
    V = np.zeros((ROWS, COLS))
    policy = {}
    rewards_map = {(r, c): 1.0 if (r == ROWS-1 and c == COLS-1) else -0.01
                   for r in range(ROWS) for c in range(COLS)}
    moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    while True:
        delta = 0
        for r in range(ROWS):
            for c in range(COLS):
                if r == ROWS-1 and c == COLS-1:
                    continue
                v = V[r, c]
                vals = []
                for a, (dr, dc) in moves.items():
                    nr, nc = max(0, min(ROWS-1, r+dr)), max(0, min(COLS-1, c+dc))
                    vals.append(rewards_map[(nr, nc)] + gamma * V[nr, nc])
                V[r, c] = max(vals)
                delta = max(delta, abs(v - V[r, c]))
        if delta < theta:
            break

    for r in range(ROWS):
        for c in range(COLS):
            if r == ROWS-1 and c == COLS-1:
                policy[(r, c)] = None
                continue
            vals = []
            for a, (dr, dc) in moves.items():
                nr, nc = max(0, min(ROWS-1, r+dr)), max(0, min(COLS-1, c+dc))
                vals.append((a, rewards_map[(nr, nc)] + gamma * V[nr, nc]))
            policy[(r, c)] = max(vals, key=lambda x: x[1])[0]
    return V, policy


def state_to_pos(state_tuple):
    idx = list(state_tuple).index(1.0)
    return divmod(idx, COLS)


def cell_center(r, c):
    return PAD + c * CELL + CELL // 2, PAD + r * CELL + CELL // 2


# ─────────────────────────────────────────────────────────────────────────────
# Canvas du plateau
# ─────────────────────────────────────────────────────────────────────────────

class GridCanvas(tk.Canvas):
    def __init__(self, parent, **kw):
        super().__init__(parent, width=CW, height=CH, bg=BG2,
                         highlightthickness=0, **kw)

    def draw_grid(self, Q=None, policy=None, opt_policy=None,
                  agent_pos=None, path=None):
        self.delete("all")
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

        for r in range(ROWS):
            for c in range(COLS):
                x0 = PAD + c * CELL
                y0 = PAD + r * CELL
                is_goal = (r == ROWS-1 and c == COLS-1)
                is_agent = (agent_pos == (r, c))
                fill = GOAL_C if is_goal else (CELL_C if (r+c) % 2 == 0 else CELL_C2)
                self.create_rectangle(x0, y0, x0+CELL, y0+CELL,
                                      fill=fill, outline=BG3, width=2)
                # Label de la case
                self.create_text(x0+7, y0+8, text=f"({r},{c})",
                                 fill=TXT_DIM, font=("Consolas", 7), anchor="nw")
                if is_goal:
                    self.create_text(x0+CELL//2, y0+CELL//2 - 10,
                                     text="GOAL", fill=BG, font=("Segoe UI", 9, "bold"))

                # Politique optimale (rouge)
                if opt_policy and not is_goal:
                    a = opt_policy.get((r, c))
                    if a is not None:
                        sym = ACTION_SYMBOLS[a]
                        cx, cy = cell_center(r, c)
                        self.create_text(cx - CELL//4, cy + CELL//4,
                                         text=sym, fill=OPT_C,
                                         font=("Segoe UI", 10))

                # Politique apprise Dyna-Q (or)
                if policy and not is_goal:
                    a = policy.get((r, c))
                    if a is not None:
                        sym = ACTION_SYMBOLS[a]
                        cx, cy = cell_center(r, c)
                        self.create_text(cx + CELL//4, cy + CELL//4,
                                         text=sym, fill=ARROW_C,
                                         font=("Segoe UI", 10, "bold"))

                # Q-values
                if Q:
                    s = tuple(np.zeros(ROWS * COLS))
                    lst = list(s); lst[r * COLS + c] = 1.0; s = tuple(lst)
                    for a, (dr, dc) in moves.items():
                        q = Q[s][a]
                        if abs(q) > 0.001:
                            cx, cy = cell_center(r, c)
                            offset = {0: (0, -CELL//3+4), 1: (0, CELL//3-4),
                                      2: (-CELL//3+4, 0), 3: (CELL//3-4, 0)}
                            ox, oy = offset[a]
                            self.create_text(cx+ox, cy+oy,
                                             text=f"{q:.2f}",
                                             fill=TXT_DIM, font=("Consolas", 7))

        # Trajectoire
        if path and len(path) > 1:
            pts = [cell_center(r, c) for (r, c) in path]
            for i in range(len(pts)-1):
                self.create_line(*pts[i], *pts[i+1], fill=PATH_C,
                                 width=3, arrow=tk.LAST, arrowshape=(8,10,4))

        # Agent
        if agent_pos:
            r, c = agent_pos
            cx, cy = cell_center(r, c)
            rad = CELL // 3 - 4
            self.create_oval(cx-rad, cy-rad, cx+rad, cy+rad,
                             fill=AGENT_C, outline="#ffffff", width=2)
            self.create_text(cx, cy, text="A", fill=BG,
                             font=("Segoe UI", 12, "bold"))


# ─────────────────────────────────────────────────────────────────────────────
# App principale
# ─────────────────────────────────────────────────────────────────────────────

class GridWorldApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GridWorld — Dyna-Q Visualisation")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(900, 620)

        self._env = GridWorld(ROWS, COLS)
        self._Q = None
        self._policy = None
        self._opt_policy = None
        self._path = []
        self._agent_pos = (0, 0)
        self._running = False
        self._trained = False
        self._rewards = []
        self._auto_job = None

        self._setup_styles()
        self._build_ui()
        self._draw()

    def _setup_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=BG2, foreground=TXT,
                    padding=[14, 7], font=("Segoe UI", 10))
        s.map("TNotebook.Tab",
              background=[("selected", BG3)],
              foreground=[("selected", GOLD)])
        s.configure("Train.Horizontal.TProgressbar",
                    troughcolor=BG3, background=GOLD, thickness=8, borderwidth=0)

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=20, pady=(14, 0))
        tk.Label(hdr, text="GridWorld 5×5", bg=BG, fg=GOLD,
                 font=("Segoe UI", 20, "bold")).pack(side="left")
        tk.Label(hdr, text="  Dyna-Q  ·  Value Iteration",
                 bg=BG, fg=TXT_DIM, font=("Segoe UI", 11)).pack(side="left", pady=4)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=20, pady=10)

        # ── Gauche : plateau ─────────────────────────────────────────────────
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", anchor="n")

        self._canvas = GridCanvas(left)
        self._canvas.pack()

        # Légende
        leg = tk.Frame(left, bg=BG)
        leg.pack(pady=(4, 0))
        for sym, col, lbl in [
            ("A", AGENT_C,  "Agent"),
            ("↑ (or)", ARROW_C, "Politique Dyna-Q"),
            ("↑ (rouge)", OPT_C, "Politique optimale"),
            ("━━", PATH_C, "Trajectoire"),
        ]:
            tk.Label(leg, text=sym + "  ", bg=BG, fg=col,
                     font=("Segoe UI", 9)).pack(side="left")
            tk.Label(leg, text=lbl + "   ", bg=BG, fg=TXT_DIM,
                     font=("Segoe UI", 9)).pack(side="left")

        # ── Droite : contrôles ───────────────────────────────────────────────
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(20, 0), anchor="n")

        # Entraînement
        train_f = tk.LabelFrame(right, text=" Entraînement Dyna-Q ",
                                bg=BG, fg=GOLD, font=("Segoe UI", 9),
                                padx=10, pady=8)
        train_f.pack(fill="x", pady=(0, 12))

        row0 = tk.Frame(train_f, bg=BG)
        row0.pack(fill="x")
        tk.Label(row0, text="Épisodes :", bg=BG, fg=TXT,
                 font=("Segoe UI", 10)).pack(side="left")
        self._ep_var = tk.IntVar(value=400)
        tk.Spinbox(row0, from_=50, to=5000, increment=50,
                   textvariable=self._ep_var, width=7,
                   bg=BG3, fg=TXT, buttonbackground=BG3,
                   font=("Segoe UI", 10), relief="flat").pack(side="left", padx=8)

        self._btn_train = tk.Button(train_f, text="▶  Entraîner",
                                    bg=GOLD, fg=BG, font=("Segoe UI", 10, "bold"),
                                    relief="flat", padx=14, pady=5, cursor="hand2",
                                    command=self._train)
        self._btn_train.pack(anchor="w", pady=(6, 0))

        self._prog = ttk.Progressbar(train_f, length=300, maximum=100,
                                     style="Train.Horizontal.TProgressbar")
        self._prog.pack(fill="x", pady=(6, 0))
        self._train_lbl = tk.Label(train_f, text="Pas encore entraîné",
                                   bg=BG, fg=TXT_DIM, font=("Segoe UI", 9))
        self._train_lbl.pack(anchor="w")

        # Lecture de la trajectoire
        play_f = tk.LabelFrame(right, text=" Lecture de la trajectoire ",
                               bg=BG, fg=GOLD, font=("Segoe UI", 9),
                               padx=10, pady=8)
        play_f.pack(fill="x", pady=(0, 12))

        row1 = tk.Frame(play_f, bg=BG)
        row1.pack(fill="x")
        tk.Label(row1, text="Vitesse (ms/pas) :", bg=BG, fg=TXT,
                 font=("Segoe UI", 10)).pack(side="left")
        self._speed_var = tk.IntVar(value=400)
        tk.Scale(row1, from_=50, to=1000, orient="horizontal",
                 variable=self._speed_var, bg=BG, fg=TXT, troughcolor=BG3,
                 highlightthickness=0, length=150).pack(side="left", padx=8)

        row2 = tk.Frame(play_f, bg=BG)
        row2.pack(fill="x", pady=(6, 0))
        self._btn_play = tk.Button(row2, text="▶  Jouer",
                                   bg=BG3, fg=GOLD, font=("Segoe UI", 10, "bold"),
                                   relief="flat", padx=12, pady=5, cursor="hand2",
                                   state="disabled", command=self._play_episode)
        self._btn_play.pack(side="left", padx=(0, 8))
        self._btn_stop = tk.Button(row2, text="■  Stop",
                                   bg=BG3, fg=TXT, font=("Segoe UI", 10),
                                   relief="flat", padx=12, pady=5, cursor="hand2",
                                   state="disabled", command=self._stop)
        self._btn_stop.pack(side="left")

        # Affichage options
        opt_f = tk.LabelFrame(right, text=" Affichage ",
                              bg=BG, fg=GOLD, font=("Segoe UI", 9),
                              padx=10, pady=8)
        opt_f.pack(fill="x", pady=(0, 12))
        self._show_q     = tk.BooleanVar(value=True)
        self._show_pol   = tk.BooleanVar(value=True)
        self._show_opt   = tk.BooleanVar(value=True)
        for var, lbl in [(self._show_q, "Valeurs Q"),
                         (self._show_pol, "Politique Dyna-Q (or)"),
                         (self._show_opt, "Politique optimale (rouge)")]:
            tk.Checkbutton(opt_f, text=lbl, variable=var,
                           bg=BG, fg=TXT, selectcolor=BG3,
                           activebackground=BG, activeforeground=GOLD,
                           font=("Segoe UI", 10),
                           command=self._draw).pack(anchor="w")

        # Stats
        self._stats_lbl = tk.Label(right, text="", bg=BG3, fg=TXT,
                                   font=("Consolas", 9), justify="left",
                                   padx=8, pady=6, anchor="w")
        self._stats_lbl.pack(fill="x", pady=(0, 0))

    def _draw(self):
        self._canvas.draw_grid(
            Q=self._Q if self._show_q.get() and self._Q else None,
            policy=self._policy if self._show_pol.get() else None,
            opt_policy=self._opt_policy if self._show_opt.get() else None,
            agent_pos=self._agent_pos,
            path=self._path,
        )

    # ── Entraînement ─────────────────────────────────────────────────────────

    def _train(self):
        if self._running:
            return
        self._btn_train.config(state="disabled")
        self._prog["value"] = 0
        self._train_lbl.config(text="Entraînement en cours…")
        episodes = self._ep_var.get()
        threading.Thread(target=self._train_thread, args=(episodes,), daemon=True).start()

    def _train_thread(self, episodes):
        env = GridWorld(ROWS, COLS)

        # Dyna-Q par blocs pour mettre à jour la barre
        Q = defaultdict(lambda: defaultdict(float))
        Model = {}
        visited = []
        rewards = []
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

        alpha, gamma, epsilon, n_planning = 0.1, 0.95, 0.15, 15

        for ep in range(episodes):
            state = tuple(env.reset())
            done = False
            total = 0
            while not done:
                actions = env.available_actions()
                if random.random() < epsilon:
                    a = random.choice(actions)
                else:
                    qs = [Q[state][a] for a in actions]
                    a = actions[int(np.argmax(qs))]
                ns_arr, r, done = env.step(a)
                ns = tuple(ns_arr)
                total += r
                next_actions = env.available_actions()
                best_next = max([Q[ns][na] for na in next_actions] or [0.0])
                Q[state][a] += alpha * (r + gamma * best_next - Q[state][a])
                Model[(state, a)] = (r, ns)
                if (state, a) not in visited:
                    visited.append((state, a))
                for _ in range(n_planning):
                    if not visited:
                        break
                    s_p, a_p = random.choice(visited)
                    r_p, ns_p = Model[(s_p, a_p)]
                    best_p = max([Q[ns_p][na] for na in env.available_actions()] or [0.0])
                    Q[s_p][a_p] += alpha * (r_p + gamma * best_p - Q[s_p][a_p])
                state = ns
            rewards.append(total)

            if (ep + 1) % max(1, episodes // 50) == 0:
                pct = (ep + 1) / episodes * 100
                self.after(0, lambda p=pct: self._prog.__setitem__("value", p))

        # Politique greedy
        policy = {}
        for row in range(ROWS):
            for col in range(COLS):
                if row == ROWS-1 and col == COLS-1:
                    policy[(row, col)] = None
                    continue
                s = [0.0] * (ROWS * COLS)
                s[row * COLS + col] = 1.0
                s = tuple(s)
                qs = [(a, Q[s][a]) for a in range(4)]
                policy[(row, col)] = max(qs, key=lambda x: x[1])[0]

        # Value iteration
        _, opt_policy = value_iteration(GridWorld(ROWS, COLS))

        self._Q = Q
        self._policy = policy
        self._opt_policy = opt_policy
        self._rewards = rewards
        self._trained = True
        self.after(0, self._on_train_done, rewards)

    def _on_train_done(self, rewards):
        self._btn_train.config(state="normal")
        self._btn_play.config(state="normal")
        self._prog["value"] = 100
        avg = sum(rewards[-50:]) / 50 if len(rewards) >= 50 else sum(rewards) / len(rewards)
        self._train_lbl.config(
            text=f"Terminé — {len(rewards)} épisodes | Récompense moy. (50 derniers) : {avg:.3f}"
        )
        self._stats_lbl.config(
            text=f"Épisodes : {len(rewards)}\n"
                 f"Récompense moy. totale   : {sum(rewards)/len(rewards):.3f}\n"
                 f"Récompense moy. (50 der) : {avg:.3f}\n"
                 f"Meilleure récompense     : {max(rewards):.3f}"
        )
        self._agent_pos = (0, 0)
        self._path = [(0, 0)]
        self._draw()

    # ── Lecture ───────────────────────────────────────────────────────────────

    def _play_episode(self):
        if self._running or not self._trained:
            return
        self._running = True
        self._btn_play.config(state="disabled")
        self._btn_stop.config(state="normal")
        self._env.reset()
        self._agent_pos = (0, 0)
        self._path = [(0, 0)]
        self._draw()
        self._auto_step()

    def _auto_step(self):
        if not self._running:
            return
        pos = self._agent_pos
        if pos == (ROWS-1, COLS-1):
            self._stop()
            return
        a = self._policy.get(pos)
        if a is None:
            self._stop()
            return
        dr, dc = {0: (-1,0), 1:(1,0), 2:(0,-1), 3:(0,1)}[a]
        nr = max(0, min(ROWS-1, pos[0]+dr))
        nc = max(0, min(COLS-1, pos[1]+dc))
        self._agent_pos = (nr, nc)
        self._path.append((nr, nc))
        self._draw()
        self._auto_job = self.after(self._speed_var.get(), self._auto_step)

    def _stop(self):
        self._running = False
        if self._auto_job:
            self.after_cancel(self._auto_job)
            self._auto_job = None
        self._btn_play.config(state="normal" if self._trained else "disabled")
        self._btn_stop.config(state="disabled")


if __name__ == "__main__":
    app = GridWorldApp()
    app.mainloop()
