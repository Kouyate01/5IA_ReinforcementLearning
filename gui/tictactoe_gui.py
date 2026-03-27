"""
╔══════════════════════════════════════════════════════════╗
║   TicTacToe vs Random — GUI Python (tkinter)            ║
║   Projet Deep Reinforcement Learning                     ║
╚══════════════════════════════════════════════════════════╝

Lancement :
    python tictactoe_gui.py

Dépendances :
    - tkinter  (inclus dans toute installation Python standard)
    - numpy    (pip install numpy)

L'interface est divisée en deux onglets :
    ① Jeu        — plateau jouable + stats session + vecteurs live
    ② Simulation — benchmark Random vs Random (parties/seconde)
"""

import tkinter as tk
from tkinter import ttk, font as tkfont
import numpy as np
import random
import time
import threading


# ══════════════════════════════════════════════════════════
#  ENVIRONNEMENT (identique à tictactoe_env.py)
# ══════════════════════════════════════════════════════════

WINNING_LINES = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    [0, 4, 8], [2, 4, 6],
]


class TicTacToeEnv:
    """
    Environnement TicTacToe (interface gym-like).

    État   : np.ndarray (9,) float32  — valeurs {-1, 0, 1}
    Action : int ∈ {0…8}              — index de la case choisie
    """

    def __init__(self):
        self.board  = np.zeros(9, dtype=np.int8)
        self.done   = False
        self.winner = None   # 1=agent, -1=random, 0=nul

    def reset(self) -> np.ndarray:
        self.board  = np.zeros(9, dtype=np.int8)
        self.done   = False
        self.winner = None
        return self._state()

    def step(self, action: int):
        """Joue l'action agent (+1), puis un coup random (-1)."""
        assert not self.done
        assert self.board[action] == 0

        self.board[action] = 1
        if self._wins(1):
            self.done, self.winner = True, 1
            return self._state(), 1.0, True, "win"
        if self._full():
            self.done, self.winner = True, 0
            return self._state(), 0.0, True, "draw"

        ai = random.choice(self.valid_actions())
        self.board[ai] = -1
        if self._wins(-1):
            self.done, self.winner = True, -1
            return self._state(), -1.0, True, "loss"
        if self._full():
            self.done, self.winner = True, 0
            return self._state(), 0.0, True, "draw"

        return self._state(), 0.0, False, "ongoing"

    def valid_actions(self):
        return [i for i in range(9) if self.board[i] == 0]

    def action_mask(self):
        return (self.board == 0).astype(np.float32)

    def winning_line(self, player):
        for line in WINNING_LINES:
            if all(self.board[i] == player for i in line):
                return line
        return None

    def _state(self):
        return self.board.copy().astype(np.float32)

    def _wins(self, p):
        return any(all(self.board[i] == p for i in l) for l in WINNING_LINES)

    def _full(self):
        return all(self.board != 0)


# ══════════════════════════════════════════════════════════
#  PALETTE & CONSTANTES
# ══════════════════════════════════════════════════════════

C = {
    "bg":       "#0e0f14",
    "surface":  "#161820",
    "panel":    "#1c1e28",
    "border":   "#2a2d3e",
    "accent":   "#e8c547",
    "x":        "#e05c5c",
    "o":        "#5ca8e0",
    "text":     "#d8daea",
    "muted":    "#6b6f8a",
    "green":    "#5ce07a",
    "win_bg":   "#2a2810",
}

CELL_SIZE  = 110   # px par case
BOARD_PAD  = 24    # marge autour du plateau
LINE_W     = 2     # épaisseur des lignes de grille


# ══════════════════════════════════════════════════════════
#  APPLICATION PRINCIPALE
# ══════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TicTacToe × RL")
        self.configure(bg=C["bg"])
        self.resizable(False, False)

        self._setup_fonts()
        self._build_ui()
        self._new_game()

    # ── Fonts ──────────────────────────────────────────────

    def _setup_fonts(self):
        self.f_title  = tkfont.Font(family="Georgia",       size=16, weight="bold",   slant="italic")
        self.f_mark   = tkfont.Font(family="Georgia",       size=46, weight="bold",   slant="italic")
        self.f_label  = tkfont.Font(family="Courier New",   size=8,  weight="bold")
        self.f_value  = tkfont.Font(family="Courier New",   size=13, weight="bold")
        self.f_small  = tkfont.Font(family="Courier New",   size=8)
        self.f_vec    = tkfont.Font(family="Courier New",   size=9,  weight="bold")
        self.f_status = tkfont.Font(family="Courier New",   size=9)
        self.f_tab    = tkfont.Font(family="Courier New",   size=9,  weight="bold")
        self.f_btn    = tkfont.Font(family="Courier New",   size=9,  weight="bold")
        self.f_sim    = tkfont.Font(family="Courier New",   size=10)
        self.f_big    = tkfont.Font(family="Courier New",   size=20, weight="bold")

    # ── Bâtir l'interface ──────────────────────────────────

    def _build_ui(self):
        # ── Header
        hdr = tk.Frame(self, bg=C["bg"], pady=12, padx=24)
        hdr.pack(fill="x")
        tk.Label(hdr, text="TicTacToe  ×  RL",
                 font=self.f_title, fg=C["accent"], bg=C["bg"]).pack(side="left")
        tk.Label(hdr, text="  Environnement Deep Reinforcement Learning",
                 font=self.f_small, fg=C["muted"], bg=C["bg"]).pack(side="left")

        sep = tk.Frame(self, bg=C["border"], height=1)
        sep.pack(fill="x")

        # ── Tabs
        tab_bar = tk.Frame(self, bg=C["surface"])
        tab_bar.pack(fill="x")
        self._tab_btns = {}
        for name, label in [("game", "▶  Jeu"), ("sim", "⏱  Simulation")]:
            b = tk.Button(tab_bar, text=label, font=self.f_tab,
                          bg=C["surface"], fg=C["muted"],
                          relief="flat", bd=0, padx=18, pady=10, cursor="hand2",
                          command=lambda n=name: self._switch_tab(n))
            b.pack(side="left")
            self._tab_btns[name] = b

        sep2 = tk.Frame(self, bg=C["border"], height=1)
        sep2.pack(fill="x")

        # ── Conteneur des tabs
        self._tab_frames = {}
        container = tk.Frame(self, bg=C["bg"])
        container.pack(fill="both", expand=True)

        self._tab_frames["game"] = self._build_game_tab(container)
        self._tab_frames["sim"]  = self._build_sim_tab(container)

        self._switch_tab("game")

    def _switch_tab(self, name):
        for n, f in self._tab_frames.items():
            f.pack_forget()
        self._tab_frames[name].pack(fill="both", expand=True)
        for n, b in self._tab_btns.items():
            if n == name:
                b.config(fg=C["accent"], bg=C["panel"])
            else:
                b.config(fg=C["muted"], bg=C["surface"])

    # ── TAB JEU ────────────────────────────────────────────

    def _build_game_tab(self, parent):
        frame = tk.Frame(parent, bg=C["bg"])

        # ─ Colonne gauche : plateau
        left = tk.Frame(frame, bg=C["bg"], padx=24, pady=20)
        left.pack(side="left", fill="y")

        # Mode
        mode_row = tk.Frame(left, bg=C["bg"])
        mode_row.pack(fill="x", pady=(0, 14))
        tk.Label(mode_row, text="MODE :", font=self.f_label,
                 fg=C["muted"], bg=C["bg"]).pack(side="left", padx=(0, 8))
        self._game_mode = tk.StringVar(value="human")
        for val, lbl in [("human", "Humain vs Random"), ("auto", "Random vs Random")]:
            rb = tk.Radiobutton(mode_row, text=lbl, variable=self._game_mode,
                                value=val, font=self.f_small,
                                fg=C["text"], bg=C["bg"],
                                selectcolor=C["panel"], activebackground=C["bg"],
                                activeforeground=C["accent"],
                                command=self._on_mode_change)
            rb.pack(side="left", padx=4)

        # Canvas plateau
        board_size = CELL_SIZE * 3 + BOARD_PAD * 2
        self._canvas = tk.Canvas(left, width=board_size, height=board_size,
                                 bg=C["panel"], highlightthickness=1,
                                 highlightbackground=C["border"])
        self._canvas.pack()
        self._canvas.bind("<Button-1>", self._on_canvas_click)
        self._canvas.bind("<Motion>",   self._on_canvas_hover)
        self._canvas.bind("<Leave>",    self._on_canvas_leave)
        self._hover_cell = -1

        # Status
        self._status_var = tk.StringVar(value="Votre tour — vous jouez X")
        status_row = tk.Frame(left, bg=C["panel"], padx=10, pady=8,
                              relief="flat", bd=1)
        status_row.pack(fill="x", pady=(10, 0))
        self._status_dot = tk.Canvas(status_row, width=10, height=10,
                                     bg=C["panel"], highlightthickness=0)
        self._status_dot.pack(side="left", padx=(0, 8))
        self._dot_oval = self._status_dot.create_oval(1, 1, 9, 9, fill=C["green"], outline="")
        tk.Label(status_row, textvariable=self._status_var,
                 font=self.f_status, fg=C["text"], bg=C["panel"]).pack(side="left")

        # Boutons
        btn_row = tk.Frame(left, bg=C["bg"])
        btn_row.pack(fill="x", pady=(10, 0))
        tk.Button(btn_row, text="Nouvelle partie", font=self.f_btn,
                  bg=C["accent"], fg=C["bg"], relief="flat", padx=14, pady=6,
                  cursor="hand2", command=self._new_game).pack(side="left", padx=(0, 8))
        self._auto_step_btn = tk.Button(btn_row, text="▶ Jouer un coup", font=self.f_btn,
                                        bg=C["panel"], fg=C["text"], relief="flat",
                                        padx=14, pady=6, cursor="hand2",
                                        command=self._auto_step)
        # affiché seulement en mode auto

        # Index hint
        hint = tk.Frame(left, bg=C["bg"])
        hint.pack(fill="x", pady=(14, 0))
        tk.Label(hint, text="INDEXATION", font=self.f_label,
                 fg=C["muted"], bg=C["bg"]).pack(anchor="w")
        idx_grid = tk.Frame(hint, bg=C["bg"])
        idx_grid.pack(anchor="w", pady=(4, 0))
        for i in range(9):
            r, c = divmod(i, 3)
            lbl = tk.Label(idx_grid, text=str(i), font=self.f_small,
                           fg=C["muted"], bg=C["bg"], width=3, relief="flat",
                           bd=1, highlightbackground=C["border"],
                           highlightthickness=1)
            lbl.grid(row=r, column=c, padx=1, pady=1, ipadx=4, ipady=4)

        # ─ Colonne droite : stats + vecteurs
        right = tk.Frame(frame, bg=C["bg"], padx=0, pady=20, width=300)
        right.pack(side="left", fill="y", padx=(0, 24))
        right.pack_propagate(False)

        # Stats
        self._stats = {"wins": 0, "losses": 0, "draws": 0}
        stats_panel = self._panel(right, "STATISTIQUES DE SESSION")
        stats_panel.pack(fill="x", pady=(0, 12))

        self._stat_vars = {}
        for key, label, color in [
            ("total",  "Parties jouées", C["text"]),
            ("wins",   "Victoires  (X)", C["green"]),
            ("losses", "Défaites   (O)", C["x"]),
            ("draws",  "Matchs nuls",    C["o"]),
        ]:
            row = tk.Frame(stats_panel, bg=C["panel"])
            row.pack(fill="x", padx=12, pady=2)
            tk.Label(row, text=label, font=self.f_small,
                     fg=C["muted"], bg=C["panel"]).pack(side="left")
            var = tk.StringVar(value="0")
            tk.Label(row, textvariable=var, font=self.f_value,
                     fg=color, bg=C["panel"]).pack(side="right")
            self._stat_vars[key] = var

        # Barres
        bar_frame = tk.Frame(stats_panel, bg=C["panel"], padx=12, pady=6)
        bar_frame.pack(fill="x", pady=(0, 4))
        self._bars = {}
        for key, label, color in [
            ("wins",   "Victoires", C["green"]),
            ("draws",  "Nuls",      C["o"]),
            ("losses", "Défaites",  C["x"]),
        ]:
            tk.Label(bar_frame, text=label, font=self.f_small,
                     fg=C["muted"], bg=C["panel"]).pack(anchor="w", pady=(4, 0))
            track = tk.Canvas(bar_frame, height=5, bg=C["bg"],
                              highlightthickness=0)
            track.pack(fill="x")
            fill = track.create_rectangle(0, 0, 0, 5, fill=color, outline="")
            self._bars[key] = (track, fill)

        # Vecteur d'état
        state_panel = self._panel(right, "ÉTAT — vecteur float32 (9,)")
        state_panel.pack(fill="x", pady=(0, 12))
        tk.Label(state_panel, text="state ∈ {−1, 0, 1}⁹",
                 font=self.f_small, fg=C["muted"], bg=C["panel"]).pack(padx=12, anchor="w")
        vec_row = tk.Frame(state_panel, bg=C["panel"], padx=12, pady=8)
        vec_row.pack(fill="x")
        self._state_cells = []
        for i in range(9):
            c = tk.Label(vec_row, text="0", font=self.f_vec,
                         fg=C["muted"], bg=C["bg"],
                         width=3, relief="flat", bd=0,
                         highlightbackground=C["border"], highlightthickness=1)
            c.grid(row=0, column=i, padx=2)
            self._state_cells.append(c)
        idx_row = tk.Frame(state_panel, bg=C["panel"], padx=12, pady=4)
        idx_row.pack(fill="x")
        for i in range(9):
            tk.Label(idx_row, text=f"i={i}", font=self.f_small,
                     fg=C["border"], bg=C["panel"], width=3).grid(row=0, column=i, padx=2)

        # Dernier coup one-hot
        action_panel = self._panel(right, "DERNIER COUP — one-hot (9,)")
        action_panel.pack(fill="x")
        tk.Label(action_panel, text="action ∈ {0…8}  →  vecteur ∈ {0,1}⁹",
                 font=self.f_small, fg=C["muted"], bg=C["panel"]).pack(padx=12, anchor="w")
        avec_row = tk.Frame(action_panel, bg=C["panel"], padx=12, pady=8)
        avec_row.pack(fill="x")
        self._action_cells = []
        for i in range(9):
            c = tk.Label(avec_row, text="—", font=self.f_vec,
                         fg=C["muted"], bg=C["bg"],
                         width=3, relief="flat", bd=0,
                         highlightbackground=C["border"], highlightthickness=1)
            c.grid(row=0, column=i, padx=2)
            self._action_cells.append(c)
        tk.Frame(action_panel, bg=C["panel"], height=8).pack()

        return frame

    # ── TAB SIMULATION ─────────────────────────────────────

    def _build_sim_tab(self, parent):
        frame = tk.Frame(parent, bg=C["bg"])

        inner = tk.Frame(frame, bg=C["bg"], padx=32, pady=24)
        inner.pack(fill="both", expand=True)

        tk.Label(inner, text="Simulation  Random vs Random",
                 font=self.f_title, fg=C["accent"], bg=C["bg"]).pack(anchor="w")
        tk.Label(inner, text="Mesure le nombre de parties simulées par seconde",
                 font=self.f_small, fg=C["muted"], bg=C["bg"]).pack(anchor="w", pady=(2, 16))

        # Config
        cfg = tk.Frame(inner, bg=C["bg"])
        cfg.pack(anchor="w", pady=(0, 16))

        tk.Label(cfg, text="Nombre de parties :", font=self.f_sim,
                 fg=C["text"], bg=C["bg"]).grid(row=0, column=0, padx=(0, 10))
        self._sim_n = tk.StringVar(value="10000")
        choices = ["1000", "10000", "50000", "100000", "500000"]
        om = ttk.Combobox(cfg, textvariable=self._sim_n, values=choices,
                          width=10, font=self.f_sim, state="readonly")
        om.grid(row=0, column=1, padx=(0, 16))

        self._run_btn = tk.Button(cfg, text="▶  Lancer", font=self.f_btn,
                                  bg=C["accent"], fg=C["bg"], relief="flat",
                                  padx=16, pady=6, cursor="hand2",
                                  command=self._run_simulation)
        self._run_btn.grid(row=0, column=2)

        # Progress
        self._prog_label = tk.StringVar(value="")
        tk.Label(inner, textvariable=self._prog_label, font=self.f_small,
                 fg=C["muted"], bg=C["bg"]).pack(anchor="w")
        self._prog_canvas = tk.Canvas(inner, height=4, bg=C["border"],
                                      highlightthickness=0)
        self._prog_canvas.pack(fill="x", pady=(2, 16))
        self._prog_fill = self._prog_canvas.create_rectangle(0, 0, 0, 4,
                                                              fill=C["accent"], outline="")

        # Résultats
        res_frame = tk.Frame(inner, bg=C["bg"])
        res_frame.pack(fill="x", pady=(0, 16))
        self._metric_vars = {}
        metrics = [
            ("gps",   "Parties / seconde", C["accent"]),
            ("wins",  "Taux de victoire",  C["green"]),
            ("losses","Taux de défaite",   C["x"]),
            ("draws", "Matchs nuls",       C["o"]),
            ("steps", "Coups / partie",    C["accent"]),
            ("dur",   "Durée (sec)",       C["text"]),
        ]
        for col, (key, label, color) in enumerate(metrics):
            card = tk.Frame(res_frame, bg=C["panel"], padx=14, pady=12,
                            relief="flat", bd=0,
                            highlightbackground=C["border"], highlightthickness=1)
            card.grid(row=0, column=col, padx=5, sticky="nsew")
            res_frame.columnconfigure(col, weight=1)
            tk.Label(card, text=label.upper(), font=self.f_label,
                     fg=C["muted"], bg=C["panel"]).pack(anchor="w")
            var = tk.StringVar(value="—")
            tk.Label(card, textvariable=var, font=self.f_big,
                     fg=color, bg=C["panel"]).pack(anchor="w", pady=(4, 0))
            self._metric_vars[key] = var

        # Terminal
        tk.Label(inner, text="LOG", font=self.f_label,
                 fg=C["muted"], bg=C["bg"]).pack(anchor="w", pady=(4, 4))
        term_frame = tk.Frame(inner, bg=C["bg"],
                              highlightbackground=C["border"], highlightthickness=1)
        term_frame.pack(fill="both", expand=True)
        self._terminal = tk.Text(term_frame, font=self.f_sim, bg="#0a0b0f",
                                 fg=C["muted"], relief="flat", bd=0,
                                 state="disabled", wrap="word", height=10,
                                 insertbackground=C["accent"])
        sb = tk.Scrollbar(term_frame, command=self._terminal.yview, bg=C["bg"])
        self._terminal.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._terminal.pack(fill="both", expand=True, padx=14, pady=10)

        # Tags couleur terminal
        self._terminal.tag_config("green",  foreground=C["green"])
        self._terminal.tag_config("yellow", foreground=C["accent"])
        self._terminal.tag_config("red",    foreground=C["x"])
        self._terminal.tag_config("blue",   foreground=C["o"])
        self._terminal.tag_config("white",  foreground=C["text"])

        self._tlog("$ ", "yellow")
        self._tlog("tictactoe_env.py  bench\n", "white")
        self._tlog("▶ Prêt. Configurez et lancez une simulation.\n", "green")

        return frame

    # ── Helpers UI ────────────────────────────────────────

    def _panel(self, parent, title):
        f = tk.Frame(parent, bg=C["panel"],
                     highlightbackground=C["border"], highlightthickness=1)
        tk.Label(f, text=title, font=self.f_label,
                 fg=C["muted"], bg=C["panel"]).pack(
            anchor="w", padx=12, pady=(8, 6))
        tk.Frame(f, bg=C["border"], height=1).pack(fill="x")
        return f

    def _tlog(self, msg, tag=None):
        self._terminal.config(state="normal")
        if tag:
            self._terminal.insert("end", msg, tag)
        else:
            self._terminal.insert("end", msg)
        self._terminal.see("end")
        self._terminal.config(state="disabled")

    # ══════════════════════════════════════════════════════
    #  LOGIQUE DU JEU
    # ══════════════════════════════════════════════════════

    def _new_game(self):
        self._env = TicTacToeEnv()
        self._state = self._env.reset()
        self._game_over = False
        self._last_action = None

        self._draw_board()
        self._update_state_vec(None)
        self._update_action_vec(None)
        self._set_status("Votre tour — vous jouez X", C["green"])

        mode = self._game_mode.get()
        if mode == "auto":
            self._auto_step_btn.pack(side="left")
            self._set_status("Mode auto — cliquez ▶ Jouer un coup", C["accent"])
        else:
            self._auto_step_btn.pack_forget()

    def _on_mode_change(self):
        self._new_game()

    def _on_canvas_click(self, event):
        if self._game_mode.get() != "human" or self._game_over:
            return
        cell = self._coords_to_cell(event.x, event.y)
        if cell is None or self._env.board[cell] != 0:
            return
        self._human_move(cell)

    def _on_canvas_hover(self, event):
        if self._game_mode.get() != "human" or self._game_over:
            self._hover_cell = -1
            self._draw_board()
            return
        cell = self._coords_to_cell(event.x, event.y)
        if cell != self._hover_cell and (cell is None or self._env.board[cell] == 0):
            self._hover_cell = cell if cell is not None else -1
            self._draw_board()

    def _on_canvas_leave(self, event):
        self._hover_cell = -1
        self._draw_board()

    def _coords_to_cell(self, x, y):
        c = (x - BOARD_PAD) // CELL_SIZE
        r = (y - BOARD_PAD) // CELL_SIZE
        if 0 <= r < 3 and 0 <= c < 3:
            return int(r * 3 + c)
        return None

    def _human_move(self, action):
        state, reward, done, result = self._env.step(action)
        self._state = state
        self._draw_board()
        self._update_state_vec(action)
        self._update_action_vec(action)

        if done:
            self._end_game(result, reward)
        else:
            self._set_status("Tour de O (random)…", C["accent"])

    def _auto_step(self):
        if self._game_over:
            self._new_game()
            return
        if not self._env.valid_actions():
            return
        action = random.choice(self._env.valid_actions())
        self._human_move(action)

    def _end_game(self, result, reward):
        self._game_over = True
        self._draw_board()  # redessine avec ligne gagnante

        if result == "win":
            self._stats["wins"] += 1
            self._set_status("✓  Victoire de X !", C["green"])
        elif result == "loss":
            self._stats["losses"] += 1
            self._set_status("✗  L'adversaire O gagne.", C["x"])
        else:
            self._stats["draws"] += 1
            self._set_status("=  Match nul.", C["o"])

        self._update_session_stats()

    def _set_status(self, text, color):
        self._status_var.set(text)
        self._status_dot.itemconfig(self._dot_oval, fill=color)

    # ── Dessin du plateau ─────────────────────────────────

    def _draw_board(self):
        cv = self._canvas
        cv.delete("all")
        size = CELL_SIZE * 3 + BOARD_PAD * 2

        # Fond
        cv.create_rectangle(0, 0, size, size, fill=C["panel"], outline="")

        board = self._env.board

        for i in range(9):
            r, c = divmod(i, 3)
            x0 = BOARD_PAD + c * CELL_SIZE
            y0 = BOARD_PAD + r * CELL_SIZE
            x1 = x0 + CELL_SIZE
            y1 = y0 + CELL_SIZE

            # Fond cellule
            bg = C["panel"]
            if i == self._hover_cell and not self._game_over:
                bg = "#1e2232"
            cv.create_rectangle(x0, y0, x1, y1, fill=bg, outline="")

            # Valeur
            v = board[i]
            if v == 1:
                cv.create_text(x0 + CELL_SIZE // 2, y0 + CELL_SIZE // 2,
                               text="X", font=self.f_mark, fill=C["x"])
            elif v == -1:
                cv.create_text(x0 + CELL_SIZE // 2, y0 + CELL_SIZE // 2,
                               text="O", font=self.f_mark, fill=C["o"])
            elif i == self._hover_cell:
                cv.create_text(x0 + CELL_SIZE // 2, y0 + CELL_SIZE // 2,
                               text="·", font=self.f_mark, fill=C["border"])

        # Grille
        for i in range(1, 3):
            x = BOARD_PAD + i * CELL_SIZE
            cv.create_line(x, BOARD_PAD, x, size - BOARD_PAD,
                           fill=C["border"], width=LINE_W)
            y = BOARD_PAD + i * CELL_SIZE
            cv.create_line(BOARD_PAD, y, size - BOARD_PAD, y,
                           fill=C["border"], width=LINE_W)

        # Ligne gagnante
        winner = self._env.winner
        if winner and winner != 0:
            line = self._env.winning_line(winner)
            if line:
                color = C["x"] if winner == 1 else C["o"]
                self._draw_win_line(line, color)

    def _draw_win_line(self, line, color):
        cv = self._canvas
        r0, c0 = divmod(line[0], 3)
        r2, c2 = divmod(line[2], 3)
        cx0 = BOARD_PAD + c0 * CELL_SIZE + CELL_SIZE // 2
        cy0 = BOARD_PAD + r0 * CELL_SIZE + CELL_SIZE // 2
        cx2 = BOARD_PAD + c2 * CELL_SIZE + CELL_SIZE // 2
        cy2 = BOARD_PAD + r2 * CELL_SIZE + CELL_SIZE // 2
        # Halo
        cv.create_line(cx0, cy0, cx2, cy2, fill=color, width=8,
                       capstyle="round")
        # Ligne centrale blanche
        cv.create_line(cx0, cy0, cx2, cy2, fill="white", width=2,
                       capstyle="round")

    # ── Vecteurs live ─────────────────────────────────────

    def _update_state_vec(self, last_action):
        board = self._env.board
        for i, cell in enumerate(self._state_cells):
            v = int(board[i])
            cell.config(text=str(v))
            if v == 1:
                cell.config(fg=C["x"],
                            highlightbackground=C["x"])
            elif v == -1:
                cell.config(fg=C["o"],
                            highlightbackground=C["o"])
            else:
                cell.config(fg=C["muted"],
                            highlightbackground=C["border"])

    def _update_action_vec(self, action):
        for i, cell in enumerate(self._action_cells):
            if action is None:
                cell.config(text="—", fg=C["muted"],
                            highlightbackground=C["border"])
            elif i == action:
                cell.config(text="1", fg=C["accent"],
                            highlightbackground=C["accent"])
            else:
                cell.config(text="0", fg=C["muted"],
                            highlightbackground=C["border"])

    # ── Stats session ─────────────────────────────────────

    def _update_session_stats(self):
        w = self._stats["wins"]
        l = self._stats["losses"]
        d = self._stats["draws"]
        t = w + l + d
        self._stat_vars["total"].set(str(t))
        self._stat_vars["wins"].set(str(w))
        self._stat_vars["losses"].set(str(l))
        self._stat_vars["draws"].set(str(d))

        if t > 0:
            self._prog_canvas.update_idletasks()
            W = self._prog_canvas.winfo_width()
            for key, val in [("wins", w), ("draws", d), ("losses", l)]:
                track, fill = self._bars[key]
                track.update_idletasks()
                tw = track.winfo_width()
                pct = val / t
                track.coords(fill, 0, 0, int(tw * pct), 5)

    # ══════════════════════════════════════════════════════
    #  SIMULATION
    # ══════════════════════════════════════════════════════

    def _run_simulation(self):
        try:
            n = int(self._sim_n.get())
        except ValueError:
            return
        self._run_btn.config(state="disabled", text="…")
        threading.Thread(target=self._sim_worker, args=(n,), daemon=True).start()

    def _sim_worker(self, n):
        self.after(0, lambda: self._prog_label.set("Simulation en cours…"))
        self.after(0, lambda: self._prog_canvas.coords(self._prog_fill, 0, 0, 0, 4))
        self.after(0, lambda: self._tlog(f"\n$ simulate(n_games={n:,})\n", "yellow"))

        wins = draws = losses = total_steps = 0
        t0 = time.perf_counter()

        CHUNK = min(5000, n)
        done = 0

        while done < n:
            batch = min(CHUNK, n - done)
            for _ in range(batch):
                b = [0] * 9
                steps = 0
                result = "draw"
                while True:
                    v1 = [i for i in range(9) if b[i] == 0]
                    if not v1: break
                    a1 = random.choice(v1)
                    b[a1] = 1; steps += 1
                    if any(all(b[i] == 1 for i in l) for l in WINNING_LINES):
                        result = "win"; break
                    v2 = [i for i in range(9) if b[i] == 0]
                    if not v2: break
                    a2 = random.choice(v2)
                    b[a2] = -1; steps += 1
                    if any(all(b[i] == -1 for i in l) for l in WINNING_LINES):
                        result = "loss"; break
                total_steps += steps
                if result == "win":   wins += 1
                elif result == "loss": losses += 1
                else:                  draws += 1

            done += batch
            pct = done / n
            elapsed_so_far = time.perf_counter() - t0
            W = self._prog_canvas.winfo_width()
            self.after(0, lambda p=pct, w=W: self._prog_canvas.coords(
                self._prog_fill, 0, 0, int(w * p), 4))

        elapsed = time.perf_counter() - t0
        gps = n / elapsed

        # Mise à jour UI depuis le thread principal
        def update():
            self._metric_vars["gps"].set(f"{gps:,.0f}")
            self._metric_vars["wins"].set(f"{wins/n*100:.1f}%")
            self._metric_vars["losses"].set(f"{losses/n*100:.1f}%")
            self._metric_vars["draws"].set(f"{draws/n*100:.1f}%")
            self._metric_vars["steps"].set(f"{total_steps/n:.2f}")
            self._metric_vars["dur"].set(f"{elapsed:.3f}")
            self._prog_label.set("Terminé !")

            self._tlog(f"✓ {n:,} parties simulées en {elapsed:.3f} s\n", "green")
            self._tlog(f"  Victoires : {wins:,} ({wins/n*100:.1f}%)\n", "white")
            self._tlog(f"  Défaites  : {losses:,} ({losses/n*100:.1f}%)\n", "white")
            self._tlog(f"  Nuls      : {draws:,} ({draws/n*100:.1f}%)\n", "white")
            self._tlog(f"  ► Parties/sec : {gps:,.0f}\n", "yellow")
            self._tlog(f"  Coups moyens  : {total_steps/n:.2f}\n", "white")

            self._run_btn.config(state="normal", text="▶  Lancer")

        self.after(0, update)


# ══════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.mainloop()
