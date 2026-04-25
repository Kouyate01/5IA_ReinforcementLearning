"""
benchmark.py — Comparaison de tous les agents entraînés sur un environnement.

Ce script charge les modèles sauvegardés et évalue chaque agent
en mode greedy (policy finale, sans exploration).
C'est ce script qui produit les tableaux du rapport.

Usage :
    python -m evaluation.benchmark --env line_world
    python -m evaluation.benchmark --env grid_world --eval_episodes 1000
    python -m evaluation.benchmark --env bobail --checkpoint 10000
"""

import os
import pickle
import json
import argparse
import time
import numpy as np
from tabulate import tabulate   # pip install tabulate


# ─────────────────────────────────────────────────────────────────────────────
# Chargement des modèles
# ─────────────────────────────────────────────────────────────────────────────

def load_tql(path: str):
    """Charge une Q-table sauvegardée et reconstruit l'agent."""
    from agents.tabular_q_learning import TabularQLearning
    with open(path, "rb") as f:
        data = pickle.load(f)
    #print(data)
    agent = TabularQLearning(
        alpha         = data["alpha"],
        gamma         = data["gamma"],
        epsilon       = 0.0,           # greedy à l'évaluation
        epsilon_min   = 0.0,
        epsilon_decay = 1.0,
    )
    agent._q_table = data["q_table"]
    return agent


def load_agent(agent_name: str, env_name: str, checkpoint: int, seed: int):
    """
    Charge un agent sauvegardé depuis saved_models/<env>/<agent>/.
    Retourne (agent, path) ou (None, None) si le fichier n'existe pas.
    """
    folder = os.path.join("saved_models", env_name, agent_name)

    if agent_name == "tql":
        path = os.path.join(folder, f"checkpoint_{checkpoint}_seed{seed}.pkl")
        if not os.path.exists(path):
            return None, None
        return load_tql(path), path

    # Ajouter les autres agents ici :
    # elif agent_name == "dql":
    #     path = os.path.join(folder, f"checkpoint_{checkpoint}_seed{seed}.pt")
    #     ...

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Évaluation d'un agent
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_agent(agent, env, n_episodes: int = 500) -> dict:
    """
    Évalue un agent en mode greedy sur n_episodes parties.
    Mesure : score moyen, longueur moyenne, temps moyen par action.
    """
    total_score = 0.0
    total_steps = 0
    total_time  = 0.0

    for _ in range(n_episodes):
        env.reset()
        steps = 0

        while not env.is_game_over():
            t0     = time.perf_counter()
            action = agent.select_action(env, greedy=True)
            total_time += time.perf_counter() - t0
            env.step(action)
            steps += 1

        total_score += env.score()
        total_steps += steps

    return {
        "mean_score":          round(total_score / n_episodes, 4),
        "mean_length":         round(total_steps / n_episodes, 2),
        "mean_action_time_ms": round((total_time / total_steps) * 1000, 4) if total_steps > 0 else 0,
    }


def evaluate_random(env, n_episodes: int = 500) -> dict:
    """Évalue l'agent Random (baseline)."""
    from agents.random_agent import RandomAgent
    agent = RandomAgent()

    total_score = 0.0
    total_steps = 0
    total_time  = 0.0

    for _ in range(n_episodes):
        env.reset()
        steps = 0
        while not env.is_game_over():
            t0     = time.perf_counter()
            action = agent.select_action(env)
            total_time += time.perf_counter() - t0
            env.step(action)
            steps += 1
        total_score += env.score()
        total_steps += steps

    return {
        "mean_score":          round(total_score / n_episodes, 4),
        "mean_length":         round(total_steps / n_episodes, 2),
        "mean_action_time_ms": round((total_time / total_steps) * 1000, 4) if total_steps > 0 else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark complet
# ─────────────────────────────────────────────────────────────────────────────

# Agents à comparer et leur format de sauvegarde
AGENT_REGISTRY = {
    "random":  None,    # pas de fichier sauvegardé, toujours disponible
    "tql":     "pkl",
    # "dql":   "pt",    # à ajouter plus tard
    # "ddql":  "pt",
    # "reinforce": "pt",
}

CHECKPOINTS = [1_000, 10_000, 100_000, 1_000_000]


def run_benchmark(env_name: str, seed: int = 42, eval_episodes: int = 500,
                  checkpoint: int = None) -> None:
    """
    Compare tous les agents disponibles sur un environnement.

    Args:
        env_name      : nom de l'environnement
        seed          : seed utilisé pour l'entraînement (pour retrouver les fichiers)
        eval_episodes : nombre de parties d'évaluation par agent
        checkpoint    : si précisé, évalue uniquement ce checkpoint
                        sinon évalue tous les checkpoints disponibles
    """
    from train import make_env
    env = make_env(env_name)

    checkpoints_to_eval = [checkpoint] if checkpoint else CHECKPOINTS

    print(f"\n{'='*65}")
    print(f"  BENCHMARK — {env_name}  |  seed={seed}  |  {eval_episodes} parties/éval")
    print(f"{'='*65}")

    # ── Résultats par checkpoint ──────────────────────────────────
    all_results = {}   # { checkpoint: { agent_name: metrics } }

    for cp in checkpoints_to_eval:
        cp_results = {}

        # Baseline Random (toujours disponible)
        cp_results["random"] = evaluate_random(env, n_episodes=eval_episodes)

        # Agents entraînés
        for agent_name, ext in AGENT_REGISTRY.items():
            if agent_name == "random" or ext is None:
                continue

            agent, path = load_agent(agent_name, env_name, cp, seed)
            if agent is None:
                # Modèle non disponible pour ce checkpoint
                cp_results[agent_name] = None
            else:
                metrics = evaluate_agent(agent, env, n_episodes=eval_episodes)
                cp_results[agent_name] = metrics

        all_results[cp] = cp_results

    # ── Affichage tableau ─────────────────────────────────────────
    _print_table(all_results, env_name)

    # ── Sauvegarde JSON ───────────────────────────────────────────
    folder = os.path.join("results", env_name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"benchmark_seed{seed}.json")

    serializable = {}
    for cp, agents in all_results.items():
        serializable[str(cp)] = {}
        for agent_name, metrics in agents.items():
            serializable[str(cp)][agent_name] = metrics if metrics else "N/A"

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\n  → Résultats sauvegardés : {path}")


def _print_table(all_results: dict, env_name: str) -> None:
    """Affiche un tableau comparatif lisible dans le terminal."""

    agent_names = list(AGENT_REGISTRY.keys())

    # ── Tableau scores ────────────────────────────────────────────
    print(f"\n  Score moyen (policy greedy, {env_name})")
    headers = ["Checkpoint"] + agent_names
    rows = []
    for cp, agents in sorted(all_results.items()):
        row = [f"{cp:>10}"]
        for name in agent_names:
            m = agents.get(name)
            row.append(f"{m['mean_score']:.3f}" if m else "  N/A ")
        rows.append(row)
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

    # ── Tableau longueurs ─────────────────────────────────────────
    print(f"\n  Longueur moyenne des parties (nombre de steps)")
    rows = []
    for cp, agents in sorted(all_results.items()):
        row = [f"{cp:>10}"]
        for name in agent_names:
            m = agents.get(name)
            row.append(f"{m['mean_length']:.1f}" if m else "  N/A ")
        rows.append(row)
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

    # ── Tableau temps par action ──────────────────────────────────
    print(f"\n  Temps moyen par action (ms)")
    rows = []
    for cp, agents in sorted(all_results.items()):
        row = [f"{cp:>10}"]
        for name in agent_names:
            m = agents.get(name)
            row.append(f"{m['mean_action_time_ms']:.4f}" if m else "  N/A ")
        rows.append(row)
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",           type=str, default="line_world",
                        choices=["line_world", "grid_world", "tictactoe", "bobail"])
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--eval_episodes", type=int, default=500)
    parser.add_argument("--checkpoint",    type=int, default=None,
                        help="Évaluer uniquement ce checkpoint (défaut: tous)")
    args = parser.parse_args()

    run_benchmark(
        env_name      = args.env,
        seed          = args.seed,
        eval_episodes = args.eval_episodes,
        checkpoint    = args.checkpoint,
    )
    