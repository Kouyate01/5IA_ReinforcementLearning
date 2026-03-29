"""
gui/bobail_gui.py
=================
Interface graphique tkinter pour l'environnement Bobail.

Règles rappel :
  • Plateau 5×5. Joueur 0 (X) en bas (ligne 4), Joueur 1 (O) en haut (ligne 0).
  • Le Bobail (B) part du centre (2,2).
  • Chaque tour = 2 phases :
      Phase 0 → déplacer le Bobail (8 directions, actions 0-7)
      Phase 1 → déplacer un de ses pions (5 pions × 8 dir, actions 8-47)
  • Victoire : amener le Bobail sur sa rangée d'arrivée
      J0 → ligne 0  |  J1 → ligne 4
    OU bloquer tous les déplacements de l'adversaire.

Modes :
  • Human vs Random  — l'humain joue X (J0), l'IA joue O (J1) au hasard
  • Human vs Human   — 2 joueurs sur le même clavier
  • Random vs Random — auto-step
Onglet Simulation  — benchmark N parties R-vs-R

Design : même palette premium que tictactoe_gui.py
  fond #0e0f14, or #e8c547, X rouge #e05c5c, O bleu #5ca8e0, Bobail violet #b08edb
"""

import sys
import os
import time
import random
import threading
import numpy as np
import tkinter as tk
from tkinter import ttk

# ── Import de l'environnement ─────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.bobail import Bobail  # noqa: E402

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = "#0e0f14"
BG2      = "#16171f"
BG3      = "#1e1f2b"
GOLD     = "#e8c547"
GOLD_DIM = "#a0893a"
X_COL    = "#e05c5c"    # Joueur 0
O_COL    = "#5ca8e0"    # Joueur 1
B_COL    = "#b08edb"    # Bobail
TXT      = "#d4d4d8"
TXT_DIM  = "#6b7280"
GRID_COL = "#2a2b3d"

# ── Géométrie du plateau ──────────────────────────────────────────────────────
CELL     = 80          # taille d'une case en px
PADDING  = 18
CANVAS_W = CELL * 5 + PADDING * 2
CANVAS_H = CELL * 5 + PADDING * 2
RADIUS   = 26          # rayon des pions
B_RADIUS = 20          # rayon du Bobail

DIR_NAMES = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]
DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1),
              (1, 0),  (1, -1), (0, -1), (-1, -1)]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def cell_xy(row: int, col: int):
    """Retourne le centre (x,y) d'une case (row,col)."""
    x = PADDING + col * CELL + CELL // 2
    y = PADDING + row * CELL + CELL // 2
    return x, y


def xy_to_cell(x: int, y: int):
    """Convertit des coords canvas en (row, col) ou None si hors grille."""
    col = (x - PADDING) // CELL
    row = (y - PADDING) // CELL
    if 0 <= row < 5 and 0 <= col < 5:
        return row, col
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Canvas du plateau
# ─────────────────────────────────────────────────────────────────────────────

class BobailCanvas(tk.Canvas):
    def __init__(self, parent, click_callback=None, **kwargs):
        super().__init__(parent, width=CANVAS_W, height=CANVAS_H,
                         bg=BG2, highlightthickness=0, **kwargs)
        self._click_cb = click_callback
        self.bind("<Button-1>", self._on_click)
        self._highlighted = set()   # cases surlignées (actions dispo)
        self._selected    = None    # pion sélectionné (row,col) en phase 2
        self._draw_grid()

    # ── Grille ───────────────────────────────────────────────────────────────

    def _draw_grid(self):
        self.delete("grid")
        for r in range(5):
            for c in range(5):
                x0 = PADDING + c * CELL
                y0 = PADDING + r * CELL
                # alternance subtile
                fill = BG2 if (r + c) % 2 == 0 else "#13141c"
                self.create_rectangle(x0, y0, x0 + CELL, y0 + CELL,
                                      fill=fill, outline=GRID_COL, width=1, tags="grid")
        # rangées d'arrivée colorées
        for c in range(5):
            # ligne 0 = zone victoire J1
            x0, y0 = PADDING + c * CELL, PADDING
            self.create_rectangle(x0, y0, x0 + CELL, y0 + CELL,
                                  fill="#0e1a28", outline=GRID_COL, width=1, tags="grid")
            # ligne 4 = zone victoire J0
            x0, y0 = PADDING + c * CELL, PADDING + 4 * CELL
            self.create_rectangle(x0, y0, x0 + CELL, y0 + CELL,
                                  fill="#1f1010", outline=GRID_COL, width=1, tags="grid")
        # étiquettes lignes/colonnes
        for i in range(5):
            x = PADDING + i * CELL + CELL // 2
            self.create_text(x, PADDING // 2, text=str(i),
                             fill=TXT_DIM, font=("Consolas", 8), tags="grid")
            y = PADDING + i * CELL + CELL // 2
            self.create_text(PADDING // 2 - 2, y, text=str(i),
                             fill=TXT_DIM, font=("Consolas", 8), tags="grid")

    # ── Clic ─────────────────────────────────────────────────────────────────

    def _on_click(self, event):
        if self._click_cb:
            cell = xy_to_cell(event.x, event.y)
            if cell is not None:
                self._click_cb(cell)

    # ── Rendu complet ─────────────────────────────────────────────────────────

    def render(self, board: np.ndarray, highlighted: set = None,
               selected=None, bobail_pos=None):
        self.delete("piece", "highlight", "select", "winmark")
        self._selected = selected

        # Surlignage des cases cibles
        if highlighted:
            for (r, c) in highlighted:
                x0 = PADDING + c * CELL + 4
                y0 = PADDING + r * CELL + 4
                self.create_rectangle(x0, y0, x0 + CELL - 8, y0 + CELL - 8,
                                      fill="", outline=GOLD, width=2,
                                      dash=(4, 3), tags="highlight")

        # Pion sélectionné
        if selected:
            sr, sc = selected
            x0 = PADDING + sc * CELL + 2
            y0 = PADDING + sr * CELL + 2
            self.create_rectangle(x0, y0, x0 + CELL - 4, y0 + CELL - 4,
                                  fill=BG3, outline=GOLD, width=3, tags="select")

        # Pièces
        for r in range(5):
            for c in range(5):
                val = board[r][c]
                if val == 0:
                    continue
                cx, cy = cell_xy(r, c)
                if val == 3:    # Bobail
                    self._draw_bobail(cx, cy)
                elif val == 1:  # J0 (X)
                    self._draw_pion(cx, cy, X_COL, "X")
                elif val == 2:  # J1 (O)
                    self._draw_pion(cx, cy, O_COL, "O")

    def _draw_pion(self, cx, cy, color, symbol):
        r = RADIUS
        self.create_oval(cx - r, cy - r, cx + r, cy + r,
                         fill=BG3, outline=color, width=3, tags="piece")
        self.create_text(cx, cy, text=symbol, fill=color,
                         font=("Segoe UI", 14, "bold"), tags="piece")

    def _draw_bobail(self, cx, cy):
        r = B_RADIUS
        # Halo
        self.create_oval(cx - r - 6, cy - r - 6, cx + r + 6, cy + r + 6,
                         fill="", outline=B_COL, width=1, dash=(3, 3), tags="piece")
        self.create_oval(cx - r, cy - r, cx + r, cy + r,
                         fill="#251e38", outline=B_COL, width=3, tags="piece")
        self.create_text(cx, cy, text="B", fill=B_COL,
                         font=("Segoe UI", 13, "bold"), tags="piece")

    def mark_winner(self, player: int):
        """Affiche une bannière de victoire."""
        msg = "🏆 Joueur X gagne !" if player == 0 else "🏆 Joueur O gagne !"
        col = X_COL if player == 0 else O_COL
        cx, cy = CANVAS_W // 2, CANVAS_H // 2
        self.create_rectangle(cx - 155, cy - 28, cx + 155, cy + 28,
                              fill=BG, outline=col, width=3, tags="winmark")
        self.create_text(cx, cy, text=msg, fill=col,
                         font=("Segoe UI", 15, "bold"), tags="winmark")

    def clear(self):
        self.delete("piece", "highlight", "select", "winmark")


# ─────────────────────────────────────────────────────────────────────────────
# Panneau vecteur d'état
# ─────────────────────────────────────────────────────────────────────────────

class StateVectorPanel(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG3, padx=8, pady=6, **kwargs)
        tk.Label(self, text="Vecteur d'état (77,) float32", bg=BG3, fg=GOLD,
                 font=("Consolas", 9, "bold")).pack(anchor="w")
        self._lbl = tk.Label(self, text="—", bg=BG3, fg=TXT,
                             font=("Consolas", 8), justify="left", wraplength=380)
        self._lbl.pack(anchor="w", pady=(2, 4))

        tk.Label(self, text="One-hot dernière action (48,)", bg=BG3, fg=GOLD_DIM,
                 font=("Consolas", 9, "bold")).pack(anchor="w")
        self._lbl_a = tk.Label(self, text="—", bg=BG3, fg=TXT_DIM,
                               font=("Consolas", 8), justify="left", wraplength=380)
        self._lbl_a.pack(anchor="w")

    def update(self, state: np.ndarray, action: int = None):
        # Affiche couche0 (J0), couche1 (J1), couche2 (Bobail)
        lines = []
        for name, offset in [("J0", 0), ("J1", 25), ("B ", 50)]:
            chunk = " ".join(f"{int(v)}" for v in state[offset:offset + 25])
            lines.append(f"{name} [{chunk}]")
        lines.append(f"cur={int(state[75])} phase={int(state[76])}")
        self._lbl.config(text="\n".join(lines))

        if action is not None:
            oh = np.zeros(48, dtype=np.int8)
            oh[action] = 1
            oh_str = "[" + " ".join(str(v) for v in oh) + "]"
            self._lbl_a.config(text=oh_str)
        else:
            self._lbl_a.config(text="—")

    def clear(self):
        self._lbl.config(text="—")
        self._lbl_a.config(text="—")


# ─────────────────────────────────────────────────────────────────────────────
# Barre de stats de session
# ─────────────────────────────────────────────────────────────────────────────

class SessionStatsBar(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG, **kwargs)
        self.wins = self.losses = self.draws = 0
        lbl = dict(bg=BG, fg=TXT, font=("Segoe UI", 9))
        big = dict(bg=BG, font=("Segoe UI", 12, "bold"))

        tk.Label(self, text="SESSION", bg=BG, fg=GOLD,
                 font=("Segoe UI", 8, "bold")).grid(row=0, column=0, columnspan=9, pady=(0, 4))

        tk.Label(self, text="✔ Victoires", **lbl).grid(row=1, column=0, sticky="w")
        self._w_lbl = tk.Label(self, text="0", fg=X_COL, **big)
        self._w_lbl.grid(row=1, column=1, padx=(4, 12))
        self._w_bar = ttk.Progressbar(self, length=80, maximum=100, style="W.Horizontal.TProgressbar")
        self._w_bar.grid(row=1, column=2, padx=(0, 14))

        tk.Label(self, text="✘ Défaites", **lbl).grid(row=1, column=3, sticky="w")
        self._l_lbl = tk.Label(self, text="0", fg=O_COL, **big)
        self._l_lbl.grid(row=1, column=4, padx=(4, 12))
        self._l_bar = ttk.Progressbar(self, length=80, maximum=100, style="L.Horizontal.TProgressbar")
        self._l_bar.grid(row=1, column=5, padx=(0, 14))

        tk.Label(self, text="= Nuls", **lbl).grid(row=1, column=6, sticky="w")
        self._d_lbl = tk.Label(self, text="0", fg=GOLD, **big)
        self._d_lbl.grid(row=1, column=7, padx=(4, 12))
        self._d_bar = ttk.Progressbar(self, length=80, maximum=100, style="D.Horizontal.TProgressbar")
        self._d_bar.grid(row=1, column=8)

    def record(self, result: str):
        if result == "win":
            self.wins += 1
        elif result == "loss":
            self.losses += 1
        else:
            self.draws += 1
        self._refresh()

    def _refresh(self):
        total = self.wins + self.losses + self.draws or 1
        self._w_lbl.config(text=str(self.wins))
        self._l_lbl.config(text=str(self.losses))
        self._d_lbl.config(text=str(self.draws))
        self._w_bar["value"] = self.wins / total * 100
        self._l_bar["value"] = self.losses / total * 100
        self._d_bar["value"] = self.draws / total * 100

    def reset(self):
        self.wins = self.losses = self.draws = 0
        self._refresh()


# ─────────────────────────────────────────────────────────────────────────────
# Onglet Jeu — logique centrale
# ─────────────────────────────────────────────────────────────────────────────

class GameTab(tk.Frame):

    MODES = ["Human vs Random", "Human vs Human", "Random vs Random"]

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG, **kwargs)
        self._env = Bobail()
        self._mode = tk.StringVar(value=self.MODES[0])
        self._status = tk.StringVar(value="")
        self._auto_job = None
        self._game_over = False
        self._last_action = None

        # État de la sélection interactive (2 phases)
        self._sel_phase     = 0     # suit env._phase
        self._sel_bobail_dir= None  # direction choisie pour Bobail (phase 0 → 1)
        self._sel_pion      = None  # (row,col) pion sélectionné (phase 1)

        self._build_ui()
        self._new_game()

    # ── UI ───────────────────────────────────────────────────────────────────

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

        # ── Gauche : plateau ─────────────────────────────────────────────────
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", anchor="n")

        self._canvas = BobailCanvas(left, click_callback=self._on_cell_click)
        self._canvas.pack()

        # Légende
        leg = tk.Frame(left, bg=BG)
        leg.pack(pady=(6, 0))
        for sym, col, lbl in [("X", X_COL, "Joueur 0 → ligne 4"),
                               ("O", O_COL, "Joueur 1 → ligne 0"),
                               ("B", B_COL, "Bobail")]:
            tk.Label(leg, text=f"{sym} ", bg=BG, fg=col,
                     font=("Segoe UI", 10, "bold")).pack(side="left")
            tk.Label(leg, text=f"{lbl}   ", bg=BG, fg=TXT_DIM,
                     font=("Segoe UI", 9)).pack(side="left")

        self._status_lbl = tk.Label(left, textvariable=self._status,
                                    bg=BG, fg=GOLD, font=("Segoe UI", 11, "bold"),
                                    height=2, width=44, wraplength=400, justify="center")
        self._status_lbl.pack(pady=6)

        tk.Button(left, text="⟳  Nouvelle Partie",
                  bg=BG3, fg=GOLD, activebackground=GOLD, activeforeground=BG,
                  font=("Segoe UI", 10, "bold"), relief="flat", padx=16, pady=6,
                  cursor="hand2", command=self._new_game).pack()

        # ── Droite : infos ───────────────────────────────────────────────────
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(20, 0), anchor="n")

        # Phase indicator
        self._phase_lbl = tk.Label(right, text="", bg=BG3, fg=B_COL,
                                   font=("Segoe UI", 10, "bold"), pady=6, padx=10)
        self._phase_lbl.pack(fill="x", pady=(0, 10))

        # Directions du Bobail (phase 0)
        self._dir_frame = tk.LabelFrame(right, text=" Phase 0 — Déplacer le Bobail ",
                                        bg=BG, fg=GOLD_DIM, font=("Segoe UI", 9),
                                        labelanchor="n")
        self._dir_frame.pack(fill="x", pady=(0, 10))
        self._dir_btns = {}
        grid3x3 = [(0, 1, "N"), (0, 2, "NE"), (1, 2, "E"), (2, 2, "SE"),
                   (2, 1, "S"), (2, 0, "SO"), (1, 0, "O"), (0, 0, "NO")]
        dk_map = {"N": 0, "NE": 1, "E": 2, "SE": 3, "S": 4, "SO": 5, "O": 6, "NO": 7}
        for gr, gc, name in grid3x3:
            d = dk_map[name]
            btn = tk.Button(self._dir_frame, text=name, width=4,
                            bg=BG3, fg=TXT, relief="flat",
                            font=("Segoe UI", 9), state="disabled",
                            command=lambda d=d: self._human_phase0(d))
            btn.grid(row=gr, column=gc, padx=2, pady=2)
            self._dir_btns[d] = btn
        # Centre vide
        tk.Label(self._dir_frame, text="B", bg=BG3, fg=B_COL,
                 font=("Segoe UI", 11, "bold"), width=4, height=1).grid(row=1, column=1)

        self._vec_panel = StateVectorPanel(right)
        self._vec_panel.pack(fill="x", pady=(0, 12))

        tk.Label(right, text="Statistiques de session", bg=BG, fg=GOLD,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self._stats = SessionStatsBar(right)
        self._stats.pack(anchor="w")
        tk.Button(right, text="Réinitialiser", bg=BG3, fg=TXT_DIM,
                  font=("Segoe UI", 9), relief="flat", cursor="hand2",
                  command=self._stats.reset).pack(anchor="w", pady=(6, 0))

    # ── Nouvelle partie ───────────────────────────────────────────────────────

    def _on_mode_change(self):
        self._cancel_auto()
        self._new_game()

    def _new_game(self):
        self._cancel_auto()
        self._env.reset()
        self._game_over = False
        self._last_action = None
        self._sel_pion = None
        self._canvas.clear()
        self._canvas._draw_grid()
        self._canvas.render(self._env._board)
        self._vec_panel.clear()

        mode = self._mode.get()
        if mode == "Random vs Random":
            self._status.set("Random vs Random — en cours…")
            self._phase_lbl.config(text="")
            self._update_dir_buttons([])
            self._schedule_auto()
        else:
            self._update_ui_state()

    def _cancel_auto(self):
        if self._auto_job:
            self.after_cancel(self._auto_job)
            self._auto_job = None

    # ── Refresh display ───────────────────────────────────────────────────────

    def _update_ui_state(self):
        env = self._env
        player = env._current_player
        phase  = env._phase
        psym   = "X" if player == 0 else "O"
        pcol   = X_COL if player == 0 else O_COL

        phase_str = ("Phase 1/2 — Déplacer le Bobail"
                     if phase == 0 else
                     "Phase 2/2 — Déplacer un pion")
        self._phase_lbl.config(text=f"Tour de {psym} · {phase_str}", fg=pcol)

        mode = self._mode.get()
        is_human_turn = (
            (mode == "Human vs Random" and player == 0) or
            mode == "Human vs Human"
        )

        if phase == 0 and is_human_turn:
            legal = env.available_actions()  # 0-7
            self._update_dir_buttons(legal)
            self._status.set(f"[{psym}] Choisissez une direction pour le Bobail")
        elif phase == 1 and is_human_turn:
            self._update_dir_buttons([])
            self._status.set(f"[{psym}] Cliquez sur un de vos pions, puis sur sa destination")
        else:
            self._update_dir_buttons([])

        # Render board avec sel
        highlights = set()
        if phase == 1 and self._sel_pion and is_human_turn:
            highlights = self._get_pion_targets(self._sel_pion)
        self._canvas.render(env._board, highlights,
                            selected=self._sel_pion,
                            bobail_pos=env._bobail_pos)

        state = env.get_state()
        self._vec_panel.update(state, self._last_action)

    def _update_dir_buttons(self, legal_dirs):
        for d, btn in self._dir_btns.items():
            if d in legal_dirs:
                btn.config(state="normal", bg=BG3, fg=GOLD)
            else:
                btn.config(state="disabled", bg=BG3, fg=TXT_DIM)

    def _get_pion_targets(self, pion_rc):
        """Retourne les cases cibles légales pour le pion sélectionné."""
        env = self._env
        pr, pc = pion_rc
        targets = set()
        pions = env._get_player_pions(env._current_player)
        if pion_rc not in pions:
            return targets
        pion_idx = pions.index(pion_rc)
        for d in range(8):
            action = 8 + pion_idx * 8 + d
            if action in env.available_actions():
                dr, dc = DIRECTIONS[d]
                targets.add((pr + dr, pc + dc))
        return targets

    # ── Interactions humain ───────────────────────────────────────────────────

    def _human_phase0(self, direction: int):
        """Bouton directionnel pour déplacer le Bobail."""
        env = self._env
        if self._game_over or env._phase != 0:
            return
        if direction not in env.available_actions():
            return
        self._last_action = direction
        _, reward, done = env.step(direction)
        if done:
            self._finish_game()
            return
        # Phase 2 : si c'est encore l'humain ou IA
        mode = self._mode.get()
        player = env._current_player
        is_human = (mode == "Human vs Random" and player == 0) or mode == "Human vs Human"
        if is_human:
            self._update_ui_state()
        else:
            self._update_ui_state()
            self.after(350, self._random_step)

    def _on_cell_click(self, cell):
        if self._game_over:
            return
        env = self._env
        mode = self._mode.get()
        player = env._current_player
        is_human = (mode == "Human vs Random" and player == 0) or mode == "Human vs Human"

        if not is_human or mode == "Random vs Random":
            return
        if env._phase != 1:
            return  # Phase 0 gérée par les boutons directionnels

        val = env._board[cell[0]][cell[1]]
        own_val = player + 1  # 1 pour J0, 2 pour J1

        # Sélectionner un pion
        if val == own_val:
            self._sel_pion = cell
            self._update_ui_state()
            return

        # Déplacer le pion sélectionné vers cette case
        if self._sel_pion is not None:
            pions = env._get_player_pions(player)
            if self._sel_pion not in pions:
                self._sel_pion = None
                self._update_ui_state()
                return
            pion_idx = pions.index(self._sel_pion)
            pr, pc = self._sel_pion
            dr, dc = cell[0] - pr, cell[1] - pc

            # Trouver la direction correspondante
            found_action = None
            for d_idx, (ddr, ddc) in enumerate(DIRECTIONS):
                if (ddr, ddc) == (dr, dc):
                    action = 8 + pion_idx * 8 + d_idx
                    if action in env.available_actions():
                        found_action = action
                    break

            if found_action is not None:
                self._sel_pion = None
                self._last_action = found_action
                _, reward, done = env.step(found_action)
                if done:
                    self._finish_game()
                    return
                # Si mode H vs R et joueur suivant = IA
                next_player = env._current_player
                next_is_ia = (mode == "Human vs Random" and next_player == 1)
                self._update_ui_state()
                if next_is_ia:
                    self.after(350, self._random_turn)
            else:
                # Clic invalide → re-sélectionner si c'est un pion
                self._sel_pion = None
                self._update_ui_state()

    # ── Tour IA random ────────────────────────────────────────────────────────

    def _random_turn(self):
        """Joue un tour complet de l'IA (phase 0 + phase 1) en Random."""
        if self._game_over:
            return
        self._random_step(then=self._after_ia_phase0)

    def _after_ia_phase0(self):
        """Après que l'IA a joué la phase 0, joue la phase 1 si pas done."""
        env = self._env
        if self._game_over or env._done:
            return
        if env._phase == 1:
            self.after(250, lambda: self._random_step(then=self._update_ui_state))
        else:
            self._update_ui_state()

    def _random_step(self, then=None):
        """Joue un seul step random puis appelle then()."""
        env = self._env
        if self._game_over or env.is_game_over():
            return
        actions = env.available_actions()
        if not actions:
            return
        action = random.choice(actions)
        self._last_action = action
        _, reward, done = env.step(action)
        self._canvas.render(env._board, bobail_pos=env._bobail_pos)
        state = env.get_state()
        self._vec_panel.update(state, action)
        if done:
            self._finish_game()
        elif then:
            then()

    # ── Mode Random vs Random ─────────────────────────────────────────────────

    def _schedule_auto(self):
        self._auto_job = self.after(200, self._auto_step)

    def _auto_step(self):
        self._auto_job = None
        if self._game_over:
            return
        env = self._env
        actions = env.available_actions()
        if not actions:
            return
        action = random.choice(actions)
        self._last_action = action
        _, _, done = env.step(action)
        state = env.get_state()
        self._canvas.render(env._board, bobail_pos=env._bobail_pos)
        self._vec_panel.update(state, action)
        if done:
            self._finish_game()
        else:
            phstr = "Phase 1 — Bobail" if env._phase == 0 else "Phase 2 — Pion"
            psym = "X" if env._current_player == 0 else "O"
            self._status.set(f"{psym} joue · {phstr}")
            self._schedule_auto()

    # ── Fin de partie ─────────────────────────────────────────────────────────

    def _finish_game(self):
        self._game_over = True
        env = self._env
        self._canvas.render(env._board)
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
        self._phase_lbl.config(text="Partie terminée", fg=GOLD)
        self._update_dir_buttons([])


# ─────────────────────────────────────────────────────────────────────────────
# Onglet Simulation
# ─────────────────────────────────────────────────────────────────────────────

class SimulationTab(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG, **kwargs)
        self._running = False
        self._thread  = None
        self._build_ui()

    def _build_ui(self):
        tk.Label(self, text="Benchmark — Random vs Random", bg=BG, fg=GOLD,
                 font=("Segoe UI", 18, "bold")).pack(anchor="w", padx=20, pady=(16, 6))

        cfg = tk.Frame(self, bg=BG2, padx=16, pady=12)
        cfg.pack(fill="x", padx=20, pady=(0, 10))

        tk.Label(cfg, text="Nombre de parties :", bg=BG2, fg=TXT,
                 font=("Segoe UI", 11)).grid(row=0, column=0, sticky="w")
        self._n_var = tk.IntVar(value=5000)
        tk.Spinbox(cfg, from_=100, to=500000, increment=500,
                   textvariable=self._n_var, width=10,
                   bg=BG3, fg=TXT, buttonbackground=BG3,
                   font=("Segoe UI", 11), relief="flat").grid(row=0, column=1, padx=10)

        self._btn_start = tk.Button(cfg, text="▶  Lancer",
                                    bg=GOLD, fg=BG, font=("Segoe UI", 11, "bold"),
                                    relief="flat", padx=16, pady=6, cursor="hand2",
                                    command=self._start)
        self._btn_start.grid(row=0, column=2, padx=10)

        self._btn_stop = tk.Button(cfg, text="■  Arrêter",
                                   bg=BG3, fg=TXT, font=("Segoe UI", 11),
                                   relief="flat", padx=16, pady=6, cursor="hand2",
                                   state="disabled", command=self._stop)
        self._btn_stop.grid(row=0, column=3)

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

    def _log_write(self, msg):
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
        self._log_write(f"[{time.strftime('%H:%M:%S')}] Lancement de {n} parties…")
        self._thread = threading.Thread(target=self._run, args=(n,), daemon=True)
        self._thread.start()

    def _stop(self):
        self._running = False

    def _run(self, n):
        env = Bobail()
        j0 = j1 = draws = 0
        t0 = time.perf_counter()
        every = max(1, n // 200)

        for i in range(n):
            if not self._running:
                break
            env.reset()
            while not env.is_game_over():
                a = random.choice(env.available_actions())
                env.step(a)
            if env._winner == 0:
                j0 += 1
            elif env._winner == 1:
                j1 += 1
            else:
                draws += 1

            if (i + 1) % every == 0:
                elapsed = time.perf_counter() - t0
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                pct = (i + 1) / n * 100
                self.after(0, self._update_ui, i + 1, speed, j0, j1, draws, pct)

        elapsed = time.perf_counter() - t0
        total = j0 + j1 + draws
        speed = total / elapsed if elapsed > 0 else 0
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
        self._prog_lbl.config(text=f"{total:,} parties terminées en {elapsed:.2f}s")
        self._update_ui(total, speed, j0, j1, draws, 100)
        self._log_write(
            f"[{time.strftime('%H:%M:%S')}] Done — {total:,} parties | "
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
        self.minsize(980, 680)
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
                            troughcolor=BG3, background=color, thickness=8, borderwidth=0)
        style.configure("TScrollbar", background=BG3, troughcolor=BG2, borderwidth=0)

    def _build_notebook(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        game_tab = GameTab(nb)
        sim_tab  = SimulationTab(nb)
        nb.add(game_tab, text="  🎮  Jeu  ")
        nb.add(sim_tab,  text="  📊  Simulation  ")


if __name__ == "__main__":
    app = BobailApp()
    app.mainloop()
