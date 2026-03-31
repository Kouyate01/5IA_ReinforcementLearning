"""
gui/visualise_lineworld_pygame.py
==================================
Visualisation tkinter de LineWorld.
Modes :
 - Humain (flèches clavier ← / →)
 - Random (aléatoire automatisé)
"""

import sys, os, random
import tkinter as tk
from tkinter import ttk

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from envs.line_world import LineWorld

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#0e0f14"
BG2     = "#16171f"
BG3     = "#1e1f2b"
GOLD    = "#e8c547"
TXT     = "#d4d4d8"
TXT_DIM = "#6b7280"
AGENT_C = "#5ca8e0"
GOOD_C  = "#4caf76"
BAD_C   = "#e05c5c"
NEUTRAL = "#1e1f2b"
PATH_C  = "#b08edb"

CW = 100
CH = 90
PAD = 30

# ── Canvas ────────────────────────────────────────────────────────────────────

class LineCanvas(tk.Canvas):
    def __init__(self, parent, n, **kw):
        self._n = n
        w = n * CW + 2 * PAD
        h = CH + 2 * PAD + 30
        super().__init__(parent, width=w, height=h, bg=BG2,
                         highlightthickness=0, **kw)
        self._cw = w
        self._ch = h

    def redraw(self, agent=None, path=None):
        self.delete("all")
        n = self._n

        for i in range(n):
            x0 = PAD + i * CW
            y0 = PAD
            is_l = (i == 0)
            is_r = (i == n - 1)
            fill = BAD_C if is_l else (GOOD_C if is_r else NEUTRAL)
            self.create_rectangle(x0, y0, x0 + CW, y0 + CH,
                                  fill=fill, outline=BG3, width=2)
            self.create_text(x0 + 7, y0 + 6, text=str(i),
                             fill=TXT_DIM, font=("Consolas", 8), anchor="nw")

            cx = x0 + CW // 2
            cy = y0 + CH // 2

            if is_l:
                self.create_text(cx, cy - 12, text="GAUCHE",
                                 fill=BG, font=("Segoe UI", 7, "bold"))
                self.create_text(cx, cy + 4, text="-1",
                                 fill=BG, font=("Segoe UI", 14, "bold"))
            elif is_r:
                self.create_text(cx, cy - 12, text="DROITE",
                                 fill=BG, font=("Segoe UI", 7, "bold"))
                self.create_text(cx, cy + 4, text="+1",
                                 fill=BG, font=("Segoe UI", 14, "bold"))

        # Trajectoire
        if path and len(path) > 1:
            ty = PAD + CH + 22
            for j in range(len(path) - 1):
                x1 = PAD + path[j] * CW + CW // 2
                x2 = PAD + path[j + 1] * CW + CW // 2
                self.create_line(x1, ty, x2, ty, fill=PATH_C,
                                 width=4, arrow=tk.LAST, arrowshape=(10, 12, 5))

        # Agent
        if agent is not None:
            cx2 = PAD + agent * CW + CW // 2
            cy2 = PAD + CH // 2
            r = 18
            self.create_oval(cx2 - r, cy2 - r, cx2 + r, cy2 + r,
                             fill=AGENT_C, outline="#ffffff", width=2)
            self.create_text(cx2, cy2, text="A", fill=BG,
                             font=("Segoe UI", 11, "bold"))

# ── App ───────────────────────────────────────────────────────────────────────

class LineWorldApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LineWorld — RL Visualisation")
        self.configure(bg=BG)
        self.resizable(True, True)

        self._N = 7
        self._env = LineWorld(self._N)
        self._agent = self._N // 2
        self._path = [self._N // 2]
        self._running = False
        self._auto_job = None
        
        # Variables UI
        self._mode      = tk.StringVar(value="Humain")
        self._n_var     = tk.IntVar(value=7)
        self._speed_var = tk.IntVar(value=400)

        self._setup_styles()
        self._build_ui()
        self._draw()
        
        # Bindings clavier (Humain)
        self.bind("<Left>", lambda e: self._human_step(0))
        self.bind("<Right>", lambda e: self._human_step(1))

    def _setup_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TRadiobutton", background=BG, foreground=TXT, font=("Segoe UI", 10))
        s.map("TRadiobutton", foreground=[("selected", GOLD)], background=[("active", BG)])

    def _build_ui(self):
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=20, pady=(14, 4))
        tk.Label(hdr, text="LineWorld", bg=BG, fg=GOLD, font=("Segoe UI", 20, "bold")).pack(side="left")
        tk.Label(hdr, text="  Environnement 1D", bg=BG, fg=TXT_DIM, font=("Segoe UI", 11)).pack(side="left")

        # Canvas
        self._cf = tk.Frame(self, bg=BG, padx=20)
        self._cf.pack(fill="x")
        self._canvas = LineCanvas(self._cf, self._N)
        self._canvas.pack()

        # Contrôles
        ctrl = tk.Frame(self, bg=BG, padx=20, pady=8)
        ctrl.pack(fill="x")

        # Configuration & Mode
        mode_f = tk.LabelFrame(ctrl, text=" Configuration & Mode ", bg=BG, fg=GOLD, font=("Segoe UI", 9), padx=10, pady=8)
        mode_f.pack(side="left", anchor="n", padx=(0, 14), fill="y")
        
        r0 = tk.Frame(mode_f, bg=BG); r0.pack(fill="x", pady=(0, 8))
        tk.Label(r0, text="N cases:", bg=BG, fg=TXT, font=("Segoe UI", 10)).pack(side="left")
        tk.Spinbox(r0, from_=3, to=15, textvariable=self._n_var, width=4,
                   bg=BG3, fg=TXT, buttonbackground=BG3, font=("Segoe UI", 10),
                   relief="flat", command=self._apply_n).pack(side="left", padx=6)

        for m in ["Humain", "Random"]:
            ttk.Radiobutton(mode_f, text=m, value=m, variable=self._mode,
                            command=self._on_mode_change).pack(anchor="w", pady=2)
                            
        self._msg_lbl = tk.Label(mode_f, text="Jouez avec ← et →", bg=BG, fg=AGENT_C, font=("Segoe UI", 9, "bold"))
        self._msg_lbl.pack(anchor="w", pady=(8, 0))

        # Contrôles de Partie
        plf = tk.LabelFrame(ctrl, text=" Contrôles de Partie ", bg=BG, fg=GOLD, font=("Segoe UI", 9), padx=10, pady=8)
        plf.pack(side="left", anchor="n", padx=(0, 14), fill="y")

        tk.Button(plf, text="Nouvelle Partie", bg=BG3, fg=TXT, font=("Segoe UI", 9), relief="flat",
                  padx=10, pady=4, cursor="hand2", command=self._new_game).pack(anchor="w", fill="x", pady=(0, 10))

        row1 = tk.Frame(plf, bg=BG); row1.pack(fill="x")
        tk.Label(row1, text="Vitesse Auto (ms):", bg=BG, fg=TXT, font=("Segoe UI", 9)).pack(side="left")
        sf = tk.Frame(plf, bg=BG); sf.pack(fill="x", pady=(2, 6))
        tk.Button(sf, text="-", width=2, bg=BG3, fg=TXT, relief="flat",
                  command=lambda: self._speed_var.set(max(50, self._speed_var.get()-50))).pack(side="left")
        self._speed_lbl = tk.Label(sf, text=str(self._speed_var.get()), bg=BG, fg=GOLD, font=("Segoe UI", 10, "bold"), width=4)
        self._speed_lbl.pack(side="left")
        tk.Button(sf, text="+", width=2, bg=BG3, fg=TXT, relief="flat",
                  command=lambda: self._speed_var.set(min(2000, self._speed_var.get()+50))).pack(side="left")
        self._speed_var.trace_add("write", lambda *_: self._speed_lbl.config(text=str(self._speed_var.get())))

        rb = tk.Frame(plf, bg=BG); rb.pack(pady=(4, 0))
        self._btn_play = tk.Button(rb, text="▶ Jouer Auto", bg=BG3, fg=GOLD, font=("Segoe UI", 9, "bold"),
                                   relief="flat", padx=10, pady=2, cursor="hand2", command=self._play, state="disabled")
        self._btn_play.pack(side="left", padx=(0, 8))
        self._btn_stop = tk.Button(rb, text="■ Stop", bg=BG3, fg=TXT, font=("Segoe UI", 9),
                                   relief="flat", padx=10, pady=2, cursor="hand2", state="disabled", command=self._stop)
        self._btn_stop.pack(side="left")

    def _apply_n(self):
        self._stop()
        n = self._n_var.get()
        if n != self._N:
            self._N = n
            self._canvas.destroy()
            self._canvas = LineCanvas(self._cf, n)
            self._canvas.pack()
            self._new_game()

    def _draw(self):
        self._canvas.redraw(agent=self._agent, path=self._path)

    def _on_mode_change(self):
        self._stop()
        m = self._mode.get()
        if m == "Humain":
            self._msg_lbl.config(text="Jouez avec ← et →", fg=AGENT_C)
            self._btn_play.config(state="disabled")
        elif m == "Random":
            self._msg_lbl.config(text="Auto aléatoire", fg=TXT_DIM)
            self._btn_play.config(state="normal" if not getattr(self._env, "_done", False) else "disabled")
            
    def _new_game(self):
        self._stop()
        self._env = LineWorld(self._N)
        self._agent = self._env._pos
        self._path = [self._agent]
        m = self._mode.get()
        if m == "Humain":
            self._msg_lbl.config(text="Jouez avec ← et →", fg=AGENT_C)
            self._btn_play.config(state="disabled")
        elif m == "Random":
            self._btn_play.config(state="normal")
            self._msg_lbl.config(text="Auto aléatoire", fg=TXT_DIM)
        self._draw()

    def _human_step(self, action):
        if self._mode.get() != "Humain" or self._running or getattr(self._env, "_done", False):
            return
        self._do_step(action)

    def _do_step(self, action):
        _, r, done = self._env.step(action)
        self._env._done = done # Set done property correctly
        self._agent = self._env._pos
        self._path.append(self._agent)
        self._draw()
        if done:
            self._stop()
            color = GOOD_C if r > 0 else BAD_C
            self._msg_lbl.config(text=f"Terminé ! (Score: {r})", fg=color)

    def _play(self):
        m = self._mode.get()
        if m == "Humain" or getattr(self._env, "_done", False):
            return
        self._running = True
        self._btn_play.config(state="disabled")
        self._btn_stop.config(state="normal")
        self._auto_step()

    def _auto_step(self):
        if not self._running or getattr(self._env, "_done", False):
            self._stop()
            return
        a = random.choice([0, 1])
        self._do_step(a)
        if not getattr(self._env, "_done", False):
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
    app = LineWorldApp()
    app.mainloop()
