"""
gui/visualise_lineworld_pygame.py
==================================
Visualisation tkinter de LineWorld avec Dyna-Q + Value Iteration.
(tkinter — pygame incompatible Python 3.14)

LineWorld : ligne de N cases, 0=terminal gauche (-1), N-1=terminal droit (+1)
"""

import sys, os, random, threading
from collections import defaultdict
import numpy as np
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
POL_C   = GOLD
OPT_C   = "#e05c5c"

CW = 100   # cell width
CH = 90    # cell height
PAD = 30   # padding


# ── Dyna-Q ─────────────────────────────────────────────────────────────────────

def dyna_q(n, episodes=500, alpha=0.1, gamma=0.95, epsilon=0.1, n_plan=10):
    Q = {s: [0.0, 0.0] for s in range(n)}
    Model = {}
    visited = []
    rewards_list = []
    env = LineWorld(size=n)

    for _ in range(episodes):
        env.reset()
        pos = env._pos
        done = False
        total = 0.0
        while not done:
            if random.random() < epsilon or Q[pos][0] == Q[pos][1]:
                a = random.randint(0, 1)
            else:
                a = 0 if Q[pos][0] >= Q[pos][1] else 1
            _, r, done = env.step(a)
            npos = env._pos
            total += r
            best_next = max(Q[npos])
            Q[pos][a] += alpha * (r + gamma * best_next - Q[pos][a])
            key = (pos, a)
            Model[key] = (r, npos)
            if key not in visited:
                visited.append(key)
            for _ in range(n_plan):
                sk, ak = random.choice(visited)
                rk, nk = Model[(sk, ak)]
                Q[sk][ak] += alpha * (rk + gamma * max(Q[nk]) - Q[sk][ak])
            pos = npos
        rewards_list.append(total)

    policy = {}
    for s in range(n):
        if s == 0 or s == n - 1:
            policy[s] = None
        else:
            policy[s] = 0 if Q[s][0] >= Q[s][1] else 1
    return Q, policy, rewards_list


def value_iteration(n, gamma=0.95, theta=1e-6):
    V = [0.0] * n
    V[0] = -1.0
    V[n - 1] = 1.0
    while True:
        delta = 0.0
        for s in range(1, n - 1):
            v_old = V[s]
            ql = (-1.0 if s - 1 == 0 else 0.0) + gamma * V[s - 1]
            qr = (1.0 if s + 1 == n - 1 else 0.0) + gamma * V[s + 1]
            V[s] = max(ql, qr)
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break
    policy = {}
    for s in range(n):
        if s == 0 or s == n - 1:
            policy[s] = None
        else:
            ql = (-1.0 if s - 1 == 0 else 0.0) + gamma * V[s - 1]
            qr = (1.0 if s + 1 == n - 1 else 0.0) + gamma * V[s + 1]
            policy[s] = 0 if ql >= qr else 1
    return V, policy


# ── Canvas ────────────────────────────────────────────────────────────────────

class LineCanvas(tk.Canvas):
    def __init__(self, parent, n, **kw):
        self._n = n
        w = n * CW + 2 * PAD
        h = CH + 2 * PAD + 50
        super().__init__(parent, width=w, height=h, bg=BG2,
                         highlightthickness=0, **kw)
        self._cw = w
        self._ch = h

    def redraw(self, Q=None, policy=None, opt_policy=None, V=None,
               agent=None, path=None, show_q=True, show_pol=True, show_opt=True):
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
            else:
                if V is not None:
                    self.create_text(cx, y0 + 12, text=f"V={V[i]:.2f}",
                                     fill=TXT_DIM, font=("Consolas", 8))
                if show_q and Q:
                    ql_v = Q[i][0]
                    qr_v = Q[i][1]
                    self.create_text(x0 + 8, y0 + CH - 16,
                                     text=f"←{ql_v:.2f}", fill=TXT_DIM,
                                     font=("Consolas", 7), anchor="w")
                    self.create_text(x0 + CW - 8, y0 + CH - 16,
                                     text=f"{qr_v:.2f}→", fill=TXT_DIM,
                                     font=("Consolas", 7), anchor="e")
                if show_opt and opt_policy:
                    a = opt_policy.get(i)
                    sym = "←" if a == 0 else ("→" if a == 1 else "")
                    if sym:
                        self.create_text(cx, cy + 18, text=sym,
                                         fill=OPT_C, font=("Segoe UI", 12))
                if show_pol and policy:
                    a = policy.get(i)
                    sym = "←" if a == 0 else ("→" if a == 1 else "")
                    if sym:
                        self.create_text(cx, cy, text=sym,
                                         fill=POL_C, font=("Segoe UI", 14, "bold"))

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

        # Légende
        items = [
            ("A", AGENT_C, "Agent"),
            ("← or", POL_C, "Pol.Dyna-Q"),
            ("← rouge", OPT_C, "Pol.optimale"),
            ("━", PATH_C, "Chemin"),
            ("━", BAD_C, "Term.-1"),
            ("━", GOOD_C, "Term.+1"),
        ]
        lx = 8
        ly = self._ch - 18
        for sym, col, lbl in items:
            txt = f"{sym} {lbl}"
            self.create_text(lx, ly, text=txt,
                             fill=col, font=("Segoe UI", 8), anchor="w")
            lx += len(txt) * 6 + 14


# ── App ───────────────────────────────────────────────────────────────────────

class LineWorldApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LineWorld — Dyna-Q Visualisation")
        self.configure(bg=BG)
        self.resizable(True, True)

        self._N         = 7
        self._Q         = None
        self._policy    = None
        self._opt_pol   = None
        self._V         = None
        self._agent     = self._N // 2
        self._path      = [self._N // 2]
        self._trained   = False
        self._running   = False
        self._auto_job  = None
        self._rewards   = []
        self._show_q    = tk.BooleanVar(value=True)
        self._show_pol  = tk.BooleanVar(value=True)
        self._show_opt  = tk.BooleanVar(value=True)
        self._n_var     = tk.IntVar(value=7)
        self._ep_var    = tk.IntVar(value=500)
        self._speed_var = tk.IntVar(value=400)

        self._setup_styles()
        self._build_ui()
        self._draw()

    def _setup_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("G.Horizontal.TProgressbar",
                    troughcolor=BG3, background=GOLD, thickness=8, borderwidth=0)

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=20, pady=(14, 4))
        tk.Label(hdr, text="LineWorld", bg=BG, fg=GOLD,
                 font=("Segoe UI", 20, "bold")).pack(side="left")
        tk.Label(hdr, text="  Dyna-Q · Value Iteration",
                 bg=BG, fg=TXT_DIM, font=("Segoe UI", 11)).pack(side="left")

        # Canvas frame
        self._cf = tk.Frame(self, bg=BG, padx=20)
        self._cf.pack(fill="x")
        self._canvas = LineCanvas(self._cf, self._N)
        self._canvas.pack()

        # Controls
        ctrl = tk.Frame(self, bg=BG, padx=20, pady=8)
        ctrl.pack(fill="x")

        # -- Paramètres --
        pf = tk.LabelFrame(ctrl, text=" Entraînement ", bg=BG, fg=GOLD,
                           font=("Segoe UI", 9), padx=10, pady=8)
        pf.pack(side="left", anchor="n", padx=(0, 14))

        r0 = tk.Frame(pf, bg=BG); r0.pack(fill="x")
        tk.Label(r0, text="N cases:", bg=BG, fg=TXT,
                 font=("Segoe UI", 10)).pack(side="left")
        tk.Spinbox(r0, from_=3, to=15, textvariable=self._n_var, width=4,
                   bg=BG3, fg=TXT, buttonbackground=BG3,
                   font=("Segoe UI", 10), relief="flat").pack(side="left", padx=6)

        r1 = tk.Frame(pf, bg=BG); r1.pack(fill="x", pady=(4, 0))
        tk.Label(r1, text="Épisodes:", bg=BG, fg=TXT,
                 font=("Segoe UI", 10)).pack(side="left")
        tk.Spinbox(r1, from_=50, to=5000, increment=50,
                   textvariable=self._ep_var, width=6,
                   bg=BG3, fg=TXT, buttonbackground=BG3,
                   font=("Segoe UI", 10), relief="flat").pack(side="left", padx=6)

        self._btn_train = tk.Button(pf, text="▶  Entraîner",
                                    bg=GOLD, fg=BG, font=("Segoe UI", 10, "bold"),
                                    relief="flat", padx=14, pady=4,
                                    cursor="hand2", command=self._train)
        self._btn_train.pack(anchor="w", pady=(8, 4))

        self._prog = ttk.Progressbar(pf, length=200, maximum=100,
                                     style="G.Horizontal.TProgressbar")
        self._prog.pack(fill="x")
        self._lbl_train = tk.Label(pf, text="Pas encore entraîné",
                                   bg=BG, fg=TXT_DIM, font=("Segoe UI", 9))
        self._lbl_train.pack(anchor="w")

        # -- Lecture --
        plf = tk.LabelFrame(ctrl, text=" Lecture ", bg=BG, fg=GOLD,
                            font=("Segoe UI", 9), padx=10, pady=8)
        plf.pack(side="left", anchor="n", padx=(0, 14))

        tk.Label(plf, text="Vitesse (ms):", bg=BG, fg=TXT,
                 font=("Segoe UI", 10)).pack(anchor="w")

        speed_frame = tk.Frame(plf, bg=BG)
        speed_frame.pack(fill="x")
        tk.Button(speed_frame, text="-", width=2, bg=BG3, fg=TXT,
                  relief="flat", font=("Segoe UI", 10),
                  command=lambda: self._speed_var.set(
                      max(50, self._speed_var.get() - 50)
                  )).pack(side="left")
        self._speed_lbl = tk.Label(speed_frame,
                                   text=str(self._speed_var.get()),
                                   bg=BG, fg=GOLD, font=("Segoe UI", 11, "bold"),
                                   width=5)
        self._speed_lbl.pack(side="left")
        tk.Button(speed_frame, text="+", width=2, bg=BG3, fg=TXT,
                  relief="flat", font=("Segoe UI", 10),
                  command=lambda: self._speed_var.set(
                      min(2000, self._speed_var.get() + 50)
                  )).pack(side="left")
        self._speed_var.trace_add("write",
                                  lambda *_: self._speed_lbl.config(
                                      text=str(self._speed_var.get())))

        rb = tk.Frame(plf, bg=BG); rb.pack(pady=(8, 0))
        self._btn_play = tk.Button(rb, text="▶  Jouer",
                                   bg=BG3, fg=GOLD, font=("Segoe UI", 10, "bold"),
                                   relief="flat", padx=10, pady=4,
                                   cursor="hand2", state="disabled",
                                   command=self._play)
        self._btn_play.pack(side="left", padx=(0, 8))
        self._btn_stop = tk.Button(rb, text="■  Stop",
                                   bg=BG3, fg=TXT, font=("Segoe UI", 10),
                                   relief="flat", padx=10, pady=4,
                                   cursor="hand2", state="disabled",
                                   command=self._stop)
        self._btn_stop.pack(side="left")

        # -- Affichage --
        af = tk.LabelFrame(ctrl, text=" Affichage ", bg=BG, fg=GOLD,
                           font=("Segoe UI", 9), padx=10, pady=8)
        af.pack(side="left", anchor="n")
        for var, lbl in [(self._show_q, "Valeurs Q"),
                         (self._show_pol, "Pol. Dyna-Q (or)"),
                         (self._show_opt, "Pol. optimale (rouge)")]:
            tk.Checkbutton(af, text=lbl, variable=var,
                           bg=BG, fg=TXT, selectcolor=BG3,
                           activebackground=BG, font=("Segoe UI", 10),
                           command=self._draw).pack(anchor="w")

        # Stats
        self._stats = tk.Label(self, text="", bg=BG3, fg=TXT,
                               font=("Consolas", 9), justify="left",
                               padx=10, pady=6, anchor="w")
        self._stats.pack(fill="x", padx=20, pady=(0, 16))

    def _draw(self):
        self._canvas.redraw(
            Q=self._Q if self._show_q.get() else None,
            policy=self._policy if self._show_pol.get() else None,
            opt_policy=self._opt_pol if self._show_opt.get() else None,
            V=self._V,
            agent=self._agent,
            path=self._path,
            show_q=self._show_q.get(),
            show_pol=self._show_pol.get(),
            show_opt=self._show_opt.get(),
        )

    def _train(self):
        n = self._n_var.get()
        if n != self._N:
            self._N = n
            self._agent = n // 2
            self._path = [n // 2]
            self._canvas.destroy()
            self._canvas = LineCanvas(self._cf, n)
            self._canvas.pack()

        self._btn_train.config(state="disabled")
        self._prog["value"] = 0
        self._lbl_train.config(text="Entraînement en cours…")
        threading.Thread(target=self._train_bg,
                         args=(n, self._ep_var.get()), daemon=True).start()

    def _train_bg(self, n, ep):
        Q, pol, rewards = dyna_q(n, episodes=ep)
        V, opt = value_iteration(n)
        self._Q       = Q
        self._policy  = pol
        self._opt_pol = opt
        self._V       = V
        self._rewards = rewards
        self._trained = True
        self._agent   = n // 2
        self._path    = [n // 2]
        self.after(0, self._train_done, rewards)

    def _train_done(self, rewards):
        self._btn_train.config(state="normal")
        self._btn_play.config(state="normal")
        self._prog["value"] = 100
        avg = sum(rewards[-50:]) / min(50, len(rewards))
        self._lbl_train.config(text=f"Terminé — {len(rewards)} ep | moy.={avg:.3f}")
        self._stats.config(
            text=f"  N={self._N}  |  Épisodes={len(rewards)}  |  "
                 f"Récompense moy. = {sum(rewards)/len(rewards):.3f}  |  "
                 f"Moy. (50 der.) = {avg:.3f}  |  Max = {max(rewards):.3f}"
        )
        self._draw()

    def _play(self):
        if self._running or not self._trained:
            return
        self._running = True
        self._btn_play.config(state="disabled")
        self._btn_stop.config(state="normal")
        self._agent = self._N // 2
        self._path  = [self._N // 2]
        self._draw()
        self._step()

    def _step(self):
        if not self._running:
            return
        pos = self._agent
        if pos == 0 or pos == self._N - 1:
            self._stop()
            return
        a = self._policy.get(pos)
        if a is None:
            self._stop()
            return
        self._agent = pos - 1 if a == 0 else pos + 1
        self._path.append(self._agent)
        self._draw()
        self._auto_job = self.after(self._speed_var.get(), self._step)

    def _stop(self):
        self._running = False
        if self._auto_job:
            self.after_cancel(self._auto_job)
            self._auto_job = None
        self._btn_play.config(state="normal" if self._trained else "disabled")
        self._btn_stop.config(state="disabled")


if __name__ == "__main__":
    app = LineWorldApp()
    app.mainloop()
