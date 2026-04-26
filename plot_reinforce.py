"""Generate plots for the REINFORCE report section."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

os.makedirs("figures_reinforce", exist_ok=True)

COLORS = {
    "REINFORCE":          "#4C72B0",
    "REINFORCE MB":       "#DD8452",
    "REINFORCE Critic":   "#55A868",
}
MARKERS = {"REINFORCE": "o", "REINFORCE MB": "s", "REINFORCE Critic": "^"}

CHECKPOINTS_100K = [1_000, 10_000, 100_000]
CHECKPOINTS_10K  = [1_000, 10_000]

# ──────────────────────────────────────────────────────────────────────────────
# 1. TicTacToe — score progression
# ──────────────────────────────────────────────────────────────────────────────
ttt_score = {
    "REINFORCE":        [0.901, 0.909, 0.915],
    "REINFORCE MB":     [0.858, 0.913, 0.939],
    "REINFORCE Critic": [0.903, 0.954, 0.987],
}
ttt_length = {
    "REINFORCE":        [3.412, 3.400, 3.374],
    "REINFORCE MB":     [3.450, 3.356, 3.386],
    "REINFORCE Critic": [3.488, 3.486, 3.164],
}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("TicTacToe — Progression de l'entraînement (100 000 épisodes)", fontsize=13, fontweight="bold")

for name, scores in ttt_score.items():
    axes[0].plot(CHECKPOINTS_100K, scores, marker=MARKERS[name], color=COLORS[name],
                 linewidth=2, markersize=8, label=name)
axes[0].set_xlabel("Épisodes d'entraînement")
axes[0].set_ylabel("Score moyen")
axes[0].set_title("Score moyen (éval. greedy, 500 ep.)")
axes[0].set_xscale("log")
axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
axes[0].set_xticks(CHECKPOINTS_100K)
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0.82, 1.01)

for name, lengths in ttt_length.items():
    axes[1].plot(CHECKPOINTS_100K, lengths, marker=MARKERS[name], color=COLORS[name],
                 linewidth=2, markersize=8, label=name)
axes[1].set_xlabel("Épisodes d'entraînement")
axes[1].set_ylabel("Longueur moyenne d'épisode")
axes[1].set_title("Longueur moyenne (éval. greedy, 500 ep.)")
axes[1].set_xscale("log")
axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
axes[1].set_xticks(CHECKPOINTS_100K)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures_reinforce/tictactoe_progression.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures_reinforce/tictactoe_progression.png")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Bobail — score + longueur
# ──────────────────────────────────────────────────────────────────────────────
bobail_score = {
    "REINFORCE":        [0.998, 0.998, 0.996],
    "REINFORCE MB":     [1.000, 1.000, 1.000],
    "REINFORCE Critic": [0.994, 1.000, 0.996],
}
bobail_length = {
    "REINFORCE":        [6.828, 7.144, 7.198],
    "REINFORCE MB":     [5.090, 4.956, 5.232],
    "REINFORCE Critic": [5.198, 4.742, 4.590],
}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Bobail — Progression de l'entraînement (100 000 épisodes)", fontsize=13, fontweight="bold")

for name, scores in bobail_score.items():
    axes[0].plot(CHECKPOINTS_100K, scores, marker=MARKERS[name], color=COLORS[name],
                 linewidth=2, markersize=8, label=name)
axes[0].set_xlabel("Épisodes d'entraînement")
axes[0].set_ylabel("Score moyen (taux de victoire)")
axes[0].set_title("Score moyen (éval. greedy, 500 ep.)")
axes[0].set_xscale("log")
axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
axes[0].set_xticks(CHECKPOINTS_100K)
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0.990, 1.002)

for name, lengths in bobail_length.items():
    axes[1].plot(CHECKPOINTS_100K, lengths, marker=MARKERS[name], color=COLORS[name],
                 linewidth=2, markersize=8, label=name)
axes[1].set_xlabel("Épisodes d'entraînement")
axes[1].set_ylabel("Longueur moyenne d'épisode")
axes[1].set_title("Longueur moyenne (éval. greedy, 500 ep.)")
axes[1].set_xscale("log")
axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
axes[1].set_xticks(CHECKPOINTS_100K)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures_reinforce/bobail_progression.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures_reinforce/bobail_progression.png")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Synthèse comparative — bar chart final score
# ──────────────────────────────────────────────────────────────────────────────
envs   = ["LineWorld\n(100K)", "GridWorld\n(10K)", "TicTacToe\n(100K)", "Bobail\n(100K)"]
agents = ["REINFORCE", "REINFORCE MB", "REINFORCE Critic"]
final_scores = {
    "REINFORCE":        [1.000, 0.930, 0.915, 0.996],
    "REINFORCE MB":     [1.000, 0.930, 0.939, 1.000],
    "REINFORCE Critic": [1.000, 0.930, 0.987, 0.996],
}

x     = np.arange(len(envs))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 5))
for i, agent in enumerate(agents):
    bars = ax.bar(x + i * width, final_scores[agent], width,
                  label=agent, color=list(COLORS.values())[i], alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, final_scores[agent]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax.set_xticks(x + width)
ax.set_xticklabels(envs, fontsize=10)
ax.set_ylabel("Score moyen final")
ax.set_title("Synthèse comparative — Score final des 3 variantes REINFORCE", fontsize=12, fontweight="bold")
ax.legend(loc="lower right")
ax.set_ylim(0.82, 1.04)
ax.grid(True, axis="y", alpha=0.3)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

plt.tight_layout()
plt.savefig("figures_reinforce/summary_bar_chart.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures_reinforce/summary_bar_chart.png")

# ──────────────────────────────────────────────────────────────────────────────
# 4. LineWorld + GridWorld — simple flat chart to show rapid convergence
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Environnements simples — Convergence des 3 variantes", fontsize=13, fontweight="bold")

# LineWorld
lw_score = {"REINFORCE": [1.0,1.0,1.0], "REINFORCE MB": [1.0,1.0,1.0], "REINFORCE Critic": [1.0,1.0,1.0]}
for name, scores in lw_score.items():
    axes[0].plot(CHECKPOINTS_100K, scores, marker=MARKERS[name], color=COLORS[name],
                 linewidth=2, markersize=8, label=name)
axes[0].set_xlabel("Épisodes d'entraînement")
axes[0].set_ylabel("Score moyen")
axes[0].set_title("LineWorld (100 000 épisodes)")
axes[0].set_xscale("log")
axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
axes[0].set_xticks(CHECKPOINTS_100K)
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0.90, 1.05)

# GridWorld
gw_score = {"REINFORCE": [0.930,0.930], "REINFORCE MB": [0.930,0.930], "REINFORCE Critic": [0.930,0.930]}
for name, scores in gw_score.items():
    axes[1].plot(CHECKPOINTS_10K, scores, marker=MARKERS[name], color=COLORS[name],
                 linewidth=2, markersize=8, label=name)
axes[1].set_xlabel("Épisodes d'entraînement")
axes[1].set_ylabel("Score moyen")
axes[1].set_title("GridWorld (10 000 épisodes)")
axes[1].set_xscale("log")
axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
axes[1].set_xticks(CHECKPOINTS_10K)
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0.85, 1.05)

plt.tight_layout()
plt.savefig("figures_reinforce/simple_envs.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures_reinforce/simple_envs.png")

print("\nTous les graphiques ont ete generes dans figures_reinforce/")
