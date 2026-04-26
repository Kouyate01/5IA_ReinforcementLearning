"""
gui/gridworld_gui.py
====================
Visualisation tkinter de GridWorld.
Modes :
 - Humain (flèches clavier ↑ ↓ ← →)
 - Agent  (agent RL chargé, auto-play)

Agents supportés : TQL, DQN, DDQN, DDQNER, DDQNPER, ExpertApprentice, PPO
"""

import sys, os, random, importlib, pickle
import numpy as np
import torch
import tkinter as tk
from tkinter import ttk

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.grid_world import GridWorld

# ─────────────────────────────────────────────────────────────────────────────
# Registre des agents
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
    """
    Lit les poids du checkpoint pour déduire (state_size, action_size).
    Évite les erreurs de size mismatch quand le modèle a été entraîné
    sur une représentation d'état différente de celle de l'env courant.

    ApprenticeNet  : fc.0.weight [128, state_size]  — dernière couche → action_size
    ActorCriticNet : shared.0.weight [128, state_size]  actor.0.weight [action_size, 128]
    """
    if interface != "act":
        return None, None
    sd = torch.load(model_path, map_location="cpu")
    if "fc.0.weight" in sd:          # ExpertApprenticeAgent
        state_size  = sd["fc.0.weight"].shape[1]
        last_w      = sorted(k for k in sd if k.endswith(".weight"))[-1]
        action_size = sd[last_w].shape[0]
    elif "shared.0.weight" in sd:    # PPOAgent
        state_size  = sd["shared.0.weight"].shape[1]
        action_size = sd["actor.0.weight"].shape[0]
    else:
        return None, None
    return state_size, action_size


def load_agent_fn(agent_type: str, model_path: str, env) -> callable:
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
                print(f"⚠️  Impossible d'inférer les tailles depuis {model_path}, fallback env")
                state_size, action_size = env.state_size, env.action_size
            else:
                print(f"  → Tailles inférées : state={state_size}, action={action_size}")
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
    agents = {"🎲 Random": None}
    if not os.path.isdir(base_dir):
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
TXT      = "#d4d4d8"
TXT_DIM  = "#6b7280"
AGENT_C  = "#5ca8e0"
GOAL_C   = "#4caf76"
LOSS_C   = "#e05c5c"
CELL_C   = "#1e1f2b"
CELL_C2  = "#191a25"
PATH_C   = "#b08edb"
BAD_C    = "#e05c5c"

CELL = 90
ROWS = 5
COLS = 5
PAD  = 24
CW   = COLS * CELL + 2 * PAD
CH   = ROWS * CELL + 2 * PAD


def state_to_pos(state):
    if state is None:
        return (0, 0)
    idx = list(state).index(1.0)
    return divmod(idx, COLS)


def cell_center(r, c):
    return PAD + c * CELL + CELL // 2, PAD + r * CELL + CELL // 2


# ── Canvas ────────────────────────────────────────────────────────────────────

class GridCanvas(tk.Canvas):
    def __init__(self, parent, **kw):
        super().__init__(parent, width=CW, height=CH,
                         bg=BG2, highlightthickness=0, **kw)

    def draw_grid(self, agent_pos=None, path=None):
        self.delete("all")
        for r in range(ROWS):
            for c in range(COLS):
                x0 = PAD + c * CELL
                y0 = PAD + r * CELL
                is_goal = (r == ROWS - 1 and c == COLS - 1)
                is_loss = (r == 0         and c == COLS - 1)
                if is_goal:
                    fill = GOAL_C
                elif is_loss:
                    fill = LOSS_C
                else:
                    fill = CELL_C if (r + c) % 2 == 0 else CELL_C2
                self.create_rectangle(x0, y0, x0 + CELL, y0 + CELL,
                                      fill=fill, outline=BG3, width=2)
                self.create_text(x0 + 7, y0 + 8, text=f"({r},{c})",
                                 fill=TXT_DIM, font=("Consolas", 7), anchor="nw")
                if is_goal:
                    self.create_text(x0 + CELL // 2, y0 + CELL // 2,
                                     text="GOAL", fill=BG, font=("Segoe UI", 12, "bold"))
                elif is_loss:
                    self.create_text(x0 + CELL // 2, y0 + CELL // 2,
                                     text="LOSS", fill=BG, font=("Segoe UI", 12, "bold"))

        if path and len(path) > 1:
            pts = [cell_center(r, c) for (r, c) in path]
            for i in range(len(pts) - 1):
                self.create_line(*pts[i], *pts[i + 1],
                                 fill=PATH_C, width=3,
                                 arrow=tk.LAST, arrowshape=(8, 10, 4))

        if agent_pos:
            r, c  = agent_pos
            cx, cy = cell_center(r, c)
            rad = CELL // 3 - 4
            self.create_oval(cx - rad, cy - rad, cx + rad, cy + rad,
                             fill=AGENT_C, outline="#ffffff", width=2)
            self.create_text(cx, cy, text="A", fill=BG, font=("Segoe UI", 12, "bold"))


# ── Application ───────────────────────────────────────────────────────────────

class GridWorldApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GridWorld — RL Visualisation")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(800, 550)

        self._env = GridWorld(ROWS, COLS)

        # Catalogue d'agents
        base = os.path.join(ROOT, "modeles_gui", "grid_world")
        self._agent_catalog  = scan_saved_models(base, self._env)
        self._current_act_fn = None

        self._agent_pos = (0, 0)
        self._path      = [(0, 0)]
        self._running   = False
        self._auto_job  = None

        self._mode      = tk.StringVar(value="Humain")
        self._speed_var = tk.IntVar(value=400)

        self._setup_styles()
        self._build_ui()
        self._new_game()

        self.bind("<Up>",    lambda e: self._human_step(0))
        self.bind("<Down>",  lambda e: self._human_step(1))
        self.bind("<Left>",  lambda e: self._human_step(2))
        self.bind("<Right>", lambda e: self._human_step(3))

    # ─────────────────────────────────────────────────────────────────────────
    # Styles
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TRadiobutton", background=BG, foreground=TXT, font=("Segoe UI", 10))
        s.map("TRadiobutton", foreground=[("selected", GOLD)], background=[("active", BG)])

    # ─────────────────────────────────────────────────────────────────────────
    # UI
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=20, pady=(14, 0))
        tk.Label(hdr, text="GridWorld 5×5", bg=BG, fg=GOLD,
                 font=("Segoe UI", 20, "bold")).pack(side="left")
        tk.Label(hdr, text="  Environnement 2D", bg=BG, fg=TXT_DIM,
                 font=("Segoe UI", 11)).pack(side="left", pady=4)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=20, pady=10)

        # Gauche : plateau
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", anchor="n")
        self._canvas = GridCanvas(left)
        self._canvas.pack()

        # Droite : contrôles
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(20, 0), anchor="n")

        # Mode
        mode_f = tk.LabelFrame(right, text=" Jeu / Mode ",
                               bg=BG, fg=GOLD, font=("Segoe UI", 9), padx=10, pady=15)
        mode_f.pack(fill="x", pady=(0, 12))
        for m in ["Humain", "Agent"]:
            ttk.Radiobutton(mode_f, text=m, value=m, variable=self._mode,
                            command=self._on_mode_change).pack(anchor="w", pady=4)

        tk.Button(mode_f, text="Nouvelle Partie",
                  bg=BG3, fg=TXT, font=("Segoe UI", 10, "bold"), relief="flat",
                  padx=12, pady=6, cursor="hand2",
                  command=self._new_game).pack(anchor="w", fill="x", pady=(15, 10))

        self._msg_lbl = tk.Label(mode_f, text="Jouez avec les flèches ↑↓←→",
                                 bg=BG, fg=AGENT_C, font=("Segoe UI", 10, "bold"))
        self._msg_lbl.pack(anchor="w")

        # Lecture auto
        play_f = tk.LabelFrame(right, text=" Lecture Auto ",
                               bg=BG, fg=GOLD, font=("Segoe UI", 9), padx=10, pady=15)
        play_f.pack(fill="x", pady=(0, 12))

        row1 = tk.Frame(play_f, bg=BG)
        row1.pack(fill="x", pady=(0, 10))
        tk.Label(row1, text="Vitesse (ms) :", bg=BG, fg=TXT,
                 font=("Segoe UI", 10)).pack(side="left")
        tk.Button(row1, text="-", width=2, bg=BG3, fg=TXT, relief="flat",
                  command=lambda: self._speed_var.set(
                      max(50, self._speed_var.get() - 50))).pack(side="left", padx=(6, 0))
        self._speed_lbl = tk.Label(row1, text=str(self._speed_var.get()),
                                   bg=BG, fg=GOLD, font=("Segoe UI", 10, "bold"), width=4)
        self._speed_lbl.pack(side="left")
        tk.Button(row1, text="+", width=2, bg=BG3, fg=TXT, relief="flat",
                  command=lambda: self._speed_var.set(
                      min(2000, self._speed_var.get() + 50))).pack(side="left")
        self._speed_var.trace_add(
            "write", lambda *_: self._speed_lbl.config(text=str(self._speed_var.get())))

        row2 = tk.Frame(play_f, bg=BG)
        row2.pack(fill="x")
        self._btn_play = tk.Button(
            row2, text="▶ Jouer",
            bg=BG3, fg=GOLD, font=("Segoe UI", 10, "bold"),
            relief="flat", padx=14, pady=6, cursor="hand2",
            command=self._play_episode, state="disabled"
        )
        self._btn_play.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self._btn_stop = tk.Button(
            row2, text="■ Stop",
            bg=BG3, fg=TXT, font=("Segoe UI", 10),
            relief="flat", padx=14, pady=6, cursor="hand2",
            state="disabled", command=self._stop
        )
        self._btn_stop.pack(side="left", fill="x", expand=True, padx=(4, 0))

        # Sélection d'agent
        agent_f = tk.LabelFrame(right, text=" Agent (GridWorld) ",
                                bg=BG, fg=GOLD, font=("Segoe UI", 9), padx=10, pady=15)
        agent_f.pack(fill="x", pady=(0, 12))

        self._agent_combo = ttk.Combobox(
            agent_f, values=list(self._agent_catalog.keys()), state="readonly"
        )
        self._agent_combo.pack(fill="x")
        self._agent_combo.set(list(self._agent_catalog.keys())[0])
        self._agent_combo.bind("<<ComboboxSelected>>", self._on_agent_select)

        self._agent_status_lbl = tk.Label(
            agent_f, text="Aucun agent chargé",
            bg=BG, fg=TXT_DIM, font=("Segoe UI", 9)
        )
        self._agent_status_lbl.pack(anchor="w", pady=(4, 0))

    # ─────────────────────────────────────────────────────────────────────────
    # Callbacks
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
            self._agent_status_lbl.config(text="❌ Échec du chargement", fg=BAD_C)
        else:
            self._current_act_fn = fn
            self._agent_status_lbl.config(text=f"✅ {label}", fg=GOLD)

    def _on_mode_change(self):
        self._stop()
        m    = self._mode.get()
        done = getattr(self._env, "_done", False)
        if m == "Humain":
            self._msg_lbl.config(text="Jouez avec les flèches ↑↓←→", fg=AGENT_C)
            self._btn_play.config(state="disabled")
        else:
            self._msg_lbl.config(text="Agent sélectionné dans le panneau →", fg=TXT_DIM)
            self._btn_play.config(state="disabled" if done else "normal")

    def _new_game(self):
        self._stop()
        self._env.reset()
        self._env._done = False
        self._agent_pos = (0, 0)
        self._path      = [(0, 0)]
        self._on_mode_change()
        self._draw()
        self.focus_set()

    def _draw(self):
        self._canvas.draw_grid(agent_pos=self._agent_pos, path=self._path)

    # ─────────────────────────────────────────────────────────────────────────
    # Steps
    # ─────────────────────────────────────────────────────────────────────────

    def _human_step(self, action):
        if self._mode.get() != "Humain" or self._running or getattr(self._env, "_done", False):
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

    def _pick_action(self) -> int:
        if self._current_act_fn is not None:
            try:
                return self._current_act_fn(self._env)
            except Exception as e:
                print(f"⚠️  Erreur agent : {e} — fallback Random")
        return random.choice(self._env.available_actions())

    def _play_episode(self):
        if self._mode.get() == "Humain":
            return
        if getattr(self._env, "_done", False):
            self._new_game()
        self._running = True
        self._btn_play.config(state="disabled")
        self._btn_stop.config(state="normal")
        self._auto_step()

    def _auto_step(self):
        if not self._running:
            return
        if getattr(self._env, "_done", False):
            self._stop()
            return
        self._do_step(self._pick_action())
        if not getattr(self._env, "_done", False):
            self._auto_job = self.after(self._speed_var.get(), self._auto_step)
        else:
            self._stop()

    def _stop(self):
        self._running = False
        if self._auto_job:
            self.after_cancel(self._auto_job)
            self._auto_job = None
        self._btn_stop.config(state="disabled")
        if self._mode.get() == "Agent" and not getattr(self._env, "_done", False):
            self._btn_play.config(state="normal")


if __name__ == "__main__":
    app = GridWorldApp()
    app.mainloop()