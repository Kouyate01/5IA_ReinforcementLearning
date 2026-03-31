"""
gui/visualise_gridworld_pygame.py
==================================
Visualisation tkinter de GridWorld.
Modes :
 - Humain (flèches clavier ↑ ↓ ← →)
 - Random (aléatoire automatisé)
"""

import sys
import os
import random
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
TXT      = "#d4d4d8"
TXT_DIM  = "#6b7280"
AGENT_C  = "#5ca8e0"
GOAL_C   = "#4caf76"
CELL_C   = "#1e1f2b"
CELL_C2  = "#191a25"
PATH_C   = "#b08edb"

CELL  = 90
ROWS  = 5
COLS  = 5
PAD   = 24
CW    = COLS * CELL + 2 * PAD
CH    = ROWS * CELL + 2 * PAD

def state_to_pos(state_tuple):
    if state_tuple is None:
        return (0, 0)
    idx = list(state_tuple).index(1.0)
    return divmod(idx, COLS)

def cell_center(r, c):
    return PAD + c * CELL + CELL // 2, PAD + r * CELL + CELL // 2

# ── Canvas ────────────────────────────────────────────────────────────────────

class GridCanvas(tk.Canvas):
    def __init__(self, parent, **kw):
        super().__init__(parent, width=CW, height=CH, bg=BG2, highlightthickness=0, **kw)

    def draw_grid(self, agent_pos=None, path=None):
        self.delete("all")

        for r in range(ROWS):
            for c in range(COLS):
                x0 = PAD + c * CELL
                y0 = PAD + r * CELL
                is_goal = (r == ROWS-1 and c == COLS-1)
                fill = GOAL_C if is_goal else (CELL_C if (r+c) % 2 == 0 else CELL_C2)
                self.create_rectangle(x0, y0, x0+CELL, y0+CELL, fill=fill, outline=BG3, width=2)
                self.create_text(x0+7, y0+8, text=f"({r},{c})", fill=TXT_DIM, font=("Consolas", 7), anchor="nw")
                if is_goal:
                    self.create_text(x0+CELL//2, y0+CELL//2, text="GOAL", fill=BG, font=("Segoe UI", 12, "bold"))

        if path and len(path) > 1:
            pts = [cell_center(r, c) for (r, c) in path]
            for i in range(len(pts)-1):
                self.create_line(*pts[i], *pts[i+1], fill=PATH_C, width=3, arrow=tk.LAST, arrowshape=(8,10,4))

        if agent_pos:
            r, c = agent_pos
            cx, cy = cell_center(r, c)
            rad = CELL // 3 - 4
            self.create_oval(cx-rad, cy-rad, cx+rad, cy+rad, fill=AGENT_C, outline="#ffffff", width=2)
            self.create_text(cx, cy, text="A", fill=BG, font=("Segoe UI", 12, "bold"))

# ── App ───────────────────────────────────────────────────────────────────────

class GridWorldApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GridWorld — RL Visualisation")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(800, 550)

        self._env = GridWorld(ROWS, COLS)
        self._path = [(0, 0)]
        self._agent_pos = (0, 0)
        self._running = False
        self._auto_job = None
        
        self._mode = tk.StringVar(value="Humain")
        self._speed_var = tk.IntVar(value=400)

        self._setup_styles()
        self._build_ui()
        self._new_game() # Initializes cleanly
        
        self.bind("<Up>",    lambda e: self._human_step(0))
        self.bind("<Down>",  lambda e: self._human_step(1))
        self.bind("<Left>",  lambda e: self._human_step(2))
        self.bind("<Right>", lambda e: self._human_step(3))

    def _setup_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TRadiobutton", background=BG, foreground=TXT, font=("Segoe UI", 10))
        s.map("TRadiobutton", foreground=[("selected", GOLD)], background=[("active", BG)])

    def _build_ui(self):
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=20, pady=(14, 0))
        tk.Label(hdr, text="GridWorld 5×5", bg=BG, fg=GOLD, font=("Segoe UI", 20, "bold")).pack(side="left")
        tk.Label(hdr, text="  Environnement 2D (Humain/Random)", bg=BG, fg=TXT_DIM, font=("Segoe UI", 11)).pack(side="left", pady=4)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=20, pady=10)

        # ── Gauche : plateau ──
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", anchor="n")
        self._canvas = GridCanvas(left)
        self._canvas.pack()

        # ── Droite : contrôles ──
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(20, 0), anchor="n")

        # Mode
        mode_f = tk.LabelFrame(right, text=" Jeu / Mode ", bg=BG, fg=GOLD, font=("Segoe UI", 9), padx=10, pady=15)
        mode_f.pack(fill="x", pady=(0, 20))
        for m in ["Humain", "Random"]:
            ttk.Radiobutton(mode_f, text=m, value=m, variable=self._mode,
                            command=self._on_mode_change).pack(anchor="w", pady=4)
                            
        tk.Button(mode_f, text="Nouvelle Partie", bg=BG3, fg=TXT, font=("Segoe UI", 10, "bold"), relief="flat",
                  padx=12, pady=6, cursor="hand2", command=self._new_game).pack(anchor="w", fill="x", pady=(15, 10))
                  
        self._msg_lbl = tk.Label(mode_f, text="Jouez avec les flèches ↑↓←→", bg=BG, fg=AGENT_C, font=("Segoe UI", 10, "bold"))
        self._msg_lbl.pack(anchor="w")

        # Lecture Auto
        play_f = tk.LabelFrame(right, text=" Lecture Auto (Random) ", bg=BG, fg=GOLD, font=("Segoe UI", 9), padx=10, pady=15)
        play_f.pack(fill="x", pady=(0, 12))
        
        row1 = tk.Frame(play_f, bg=BG); row1.pack(fill="x", pady=(0, 10))
        tk.Label(row1, text="Vitesse (ms) :", bg=BG, fg=TXT, font=("Segoe UI", 10)).pack(side="left")
        tk.Button(row1, text="-", width=2, bg=BG3, fg=TXT, relief="flat",
                  command=lambda: self._speed_var.set(max(50, self._speed_var.get()-50))).pack(side="left", padx=(6,0))
        self._btn_p_lbl = tk.Label(row1, text=str(self._speed_var.get()), bg=BG, fg=GOLD, font=("Segoe UI", 10, "bold"), width=4)
        self._btn_p_lbl.pack(side="left")
        tk.Button(row1, text="+", width=2, bg=BG3, fg=TXT, relief="flat",
                  command=lambda: self._speed_var.set(min(2000, self._speed_var.get()+50))).pack(side="left")
        self._speed_var.trace_add("write", lambda *_: self._btn_p_lbl.config(text=str(self._speed_var.get())))

        row2 = tk.Frame(play_f, bg=BG); row2.pack(fill="x")
        self._btn_play = tk.Button(row2, text="▶ Jouer", bg=BG3, fg=GOLD, font=("Segoe UI", 10, "bold"),
                                   relief="flat", padx=14, pady=6, cursor="hand2", command=self._play_episode, state="disabled")
        self._btn_play.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self._btn_stop = tk.Button(row2, text="■ Stop", bg=BG3, fg=TXT, font=("Segoe UI", 10),
                                   relief="flat", padx=14, pady=6, cursor="hand2", state="disabled", command=self._stop)
        self._btn_stop.pack(side="left", fill="x", expand=True, padx=(4, 0))

    def _draw(self):
        self._canvas.draw_grid(agent_pos=self._agent_pos, path=self._path)

    def _on_mode_change(self):
        self._stop()
        m = self._mode.get()
        is_done = getattr(self._env, "_done", False)
        if m == "Humain":
            self._msg_lbl.config(text="Jouez avec les flèches ↑↓←→", fg=AGENT_C)
            self._btn_play.config(state="disabled")
        elif m == "Random":
            self._msg_lbl.config(text="Auto aléatoire", fg=TXT_DIM)
            self._btn_play.config(state="normal" if not is_done else "disabled")

    def _new_game(self):
        self._stop()
        self._env.reset()
        self._env._done = False
        self._agent_pos = (0, 0)
        self._path = [(0, 0)]
        self._on_mode_change()
        self._draw()
        self.focus_set()

    # ── Contrôle step ──
    def _human_step(self, action):
        if self._mode.get() != "Humain" or self._running or getattr(self._env, "_done", False):
            return
        
        s = self._env.get_state()
        if s is not None and s[ROWS*COLS - 1] == 1.0:
            return
            
        self._do_step(action)

    def _do_step(self, action):
        _, r, done = self._env.step(action)
        self._env._done = done
        self._agent_pos = state_to_pos(self._env.get_state())
        self._path.append(self._agent_pos)
        self._draw()
        if done:
            self._stop()
            color = GOAL_C if r > 0 else BAD_C
            self._msg_lbl.config(text=f"Terminé ! (Score: {r:.2f})", fg=color)

    # ── Lecture Auto ──
    def _play_episode(self):
        m = self._mode.get()
        if m == "Humain":
            return
            
        if getattr(self._env, "_done", False) or self._agent_pos == (ROWS-1, COLS-1):
            self._new_game()
            
        self._running = True
        self._btn_play.config(state="disabled")
        self._btn_stop.config(state="normal")
        self._auto_step()

    def _auto_step(self):
        if not self._running:
            return
        
        if getattr(self._env, "_done", False) or self._agent_pos == (ROWS-1, COLS-1):
            self._stop()
            self._msg_lbl.config(text=f"Terminé !", fg=GOAL_C)
            return
            
        a = random.choice(self._env.available_actions())
        self._do_step(a)
        
        if getattr(self._env, "_done", False) or self._agent_pos == (ROWS-1, COLS-1):
            self._stop()
            self._msg_lbl.config(text=f"Terminé !", fg=GOAL_C)
            return
            
        self._auto_job = self.after(self._speed_var.get(), self._auto_step)

    def _stop(self):
        self._running = False
        if self._auto_job:
            self.after_cancel(self._auto_job)
            self._auto_job = None
        self._btn_stop.config(state="disabled")
        if self._mode.get() == "Random" and not getattr(self._env, "_done", False):
            self._btn_play.config(state="normal")

if __name__ == "__main__":
    app = GridWorldApp()
    app.mainloop()
