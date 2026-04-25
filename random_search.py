"""
random_search.py — Random search sur les hyperparamètres des agents DQN

Usage :
    python random_search.py --agent dqn   --env tictactoe --n_runs 15 --episodes 10000
    python random_search.py --agent ddqn  --env tictactoe --n_runs 15 --episodes 10000
    python random_search.py --agent ddqner  --env bobail  --n_runs 15 --episodes 10000
    python random_search.py --agent ddqnper --env bobail  --n_runs 15 --episodes 10000

Le script :
    1. Tire `n_runs` configs aléatoirement dans l'espace défini par agent
    2. Entraîne chaque config sur 1 seed (seed=0) pendant `episodes` épisodes
    3. Évalue en mode greedy sur 500 parties après l'entraînement
    4. Sauvegarde les résultats dans results/<env>/<agent>/random_search.json
    5. Affiche le classement final
"""

import argparse
import json
import os
import random
import time
import numpy as np
import torch
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Espaces de recherche par agent
# Seuls 2 paramètres sensibles sont explorés — les autres restent fixes.
# ─────────────────────────────────────────────────────────────────────────────

def sample_config(agent_name: str) -> dict:
    """
    Tire une configuration aléatoire pour l'agent donné.

    Stratégie :
        - Les paramètres fixes sont issus de la baseline validée.
        - Les 2 paramètres sensibles sont tirés dans un espace log-uniforme
          (pour lr, buffer_size) ou uniforme (pour les autres).

    Log-uniforme pour lr :
        On tire x ~ Uniform(log(lo), log(hi)) puis lr = exp(x)
        → couvre plusieurs ordres de grandeur de façon équilibrée.
        Ex : Uniform(log(1e-4), log(1e-2)) donne autant de chances à
             [1e-4, 1e-3] qu'à [1e-3, 1e-2].
    """

    if agent_name == "dqn":
        # Paramètres fixes
        base = {
            "gamma":         0.99,
            "epsilon":       1.0,
            "epsilon_min":   0.01,
            "batch_size":    64,
            "buffer_size":   10000,
            "hidden_sizes":  [128, 128],
            "target_update_freq": 1000,
        }
        # Paramètres explorés
        base["lr"]            =  _log_uniform(1e-5, 1e-3) #_log_uniform(1e-4, 1e-2)
        base["epsilon_decay"] = _uniform(0.990, 0.999)
        return base

    elif agent_name == "ddqn":
        base = {
            "gamma":         0.99,
            "epsilon":       1.0,
            "epsilon_min":   0.01,
            "batch_size":    64,
            "buffer_size":   10_000,
            "hidden_sizes":  [128, 128],
            "epsilon_decay": 0.995,
        }
        base["lr"]                 = _log_uniform(1e-4, 1e-2)
        base["target_update_freq"] = _int_uniform(5, 50)
        return base

    elif agent_name == "ddqner":
        base = {
            "gamma":              0.99,
            "epsilon":            1.0,
            "epsilon_min":        0.01,
            "epsilon_decay":      0.998,
            "batch_size":         128,
            "hidden_sizes":       [128, 128],
            "target_update_freq": 10,
            "warmup_steps":       2_000,
            "updates_per_step":   1,
        }
        base["lr"]          = _log_uniform(1e-4, 1e-2)
        base["buffer_size"] = _log_int_uniform(10_000, 200_000)
        return base

    elif agent_name == "ddqnper":
        base = {
            "gamma":              0.99,
            "epsilon":            1.0,
            "epsilon_min":        0.01,
            "epsilon_decay":      0.998,
            "batch_size":         128,
            "buffer_size":        50_000,
            "hidden_sizes":       [128, 128],
            "target_update_freq": 10,
            "warmup_steps":       2_000,
            "lr":                 5e-4,
            "per_beta_max":       1.0,
            "per_eps":            1e-6,
        }
        base["per_alpha"] = _uniform(0.3, 0.8)   # force de priorisation
        base["per_beta"]  = _uniform(0.2, 0.6)   # correction IS initiale
        return base

    raise ValueError(f"Agent inconnu : {agent_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires de tirage
# ─────────────────────────────────────────────────────────────────────────────

def _log_uniform(lo: float, hi: float) -> float:
    """Tirage log-uniforme dans [lo, hi]."""
    return float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))

def _uniform(lo: float, hi: float) -> float:
    """Tirage uniforme dans [lo, hi]."""
    return float(np.random.uniform(lo, hi))

def _int_uniform(lo: int, hi: int) -> int:
    """Tirage entier uniforme dans [lo, hi]."""
    return int(np.random.randint(lo, hi + 1))

def _log_int_uniform(lo: int, hi: int) -> int:
    """Tirage entier log-uniforme dans [lo, hi]."""
    return int(np.exp(np.random.uniform(np.log(lo), np.log(hi))))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (réutilisés depuis train.py)
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(env_name: str):
    if env_name == "line_world":
        from envs.line_world import LineWorld
        return LineWorld()
    elif env_name == "grid_world":
        from envs.grid_world import GridWorld
        return GridWorld()
    elif env_name == "tictactoe":
        from envs.tictactoe import TicTacToe
        return TicTacToe()
    elif env_name == "bobail":
        from envs.bobail import Bobail
        return Bobail()
    raise ValueError(f"Environnement inconnu : {env_name}")


def make_agent(agent_name: str, config: dict):
    if agent_name == "dqn":
        from agents.dqn import DeepQLearning
        return DeepQLearning.from_config(config)
    elif agent_name == "ddqn":
        from agents.ddqn import DoubleDeepQLearning
        return DoubleDeepQLearning.from_config(config)
    elif agent_name == "ddqner":
        from agents.ddqner import DoubleDeepQLearningWithExperienceReplay
        return DoubleDeepQLearningWithExperienceReplay.from_config(config)
    elif agent_name == "ddqnper":
        from agents.ddqnper import DoubleDeepQLearningWithPrioritizedExperienceReplay
        return DoubleDeepQLearningWithPrioritizedExperienceReplay.from_config(config)
    raise ValueError(f"Agent inconnu : {agent_name}")

# ─────────────────────────────────────────────────────────────────────────────
# Évaluation de UNE config sur 3–5 seeds
# ─────────────────────────────────────────────────────────────────────────────

def eval_config_on_seeds(
    agent_name: str,
    env_name: str,
    config: dict,
    n_episodes: int,
    eval_episodes: int,
    seeds: list,
) -> dict:
    """
    Pour une config donnée, lance 3–5 seeds et retourne :
        mean_score, std_score, mean_train_time (s), mean_action_time_ms
    """
    results = {
        "seed_results": [],
        "mean_score": 0.0,
        "std_score": 0.0,
        "mean_train_time": 0.0,
        "mean_action_time_ms": 0.0,
    }

    scores = []
    train_times = []
    action_times = []

    print(f"  → {len(seeds)} seeds pour la même config")

    for i, seed in enumerate(seeds):
        set_seed(seed)
        config_seed = {k: v for k, v in config.items()}  # copie légère

        env = make_env(env_name)
        agent = make_agent(agent_name, config_seed)

        # Entraînement court
        t0 = time.perf_counter()
        agent.train(
            env=env,
            n_episodes=n_episodes,
            eval_episodes=0,          # pas d'éval intermédiaire
            log_dir=f"runs/_rs_/{agent_name}_{env_name}_config_seed{seed}",
            log_every=n_episodes,     # log uniquement à la fin
            save_callback=None,
        )
        train_time = time.perf_counter() - t0

        # Évaluation greedy
        metrics = agent.evaluate(env, n_episodes=eval_episodes)

        scores.append(metrics["mean_score"])
        train_times.append(train_time)
        action_times.append(metrics["mean_action_time_ms"])

        results["seed_results"].append({
            "seed": seed,
            "score": metrics["mean_score"],
            "length": metrics["mean_length"],
            "action_ms": metrics["mean_action_time_ms"],
            "train_s": train_time,
        })

    results["mean_score"] = float(np.mean(scores))
    results["std_score"] = float(np.std(scores))
    results["mean_train_time"] = float(np.mean(train_times))
    results["mean_action_time_ms"] = float(np.mean(action_times))

    return results

# ─────────────────────────────────────────────────────────────────────────────
# Fonction principale de random search
# ─────────────────────────────────────────────────────────────────────────────
def run_random_search(
    agent_name: str,
    env_name: str,
    n_runs: int = 15,
    n_episodes: int = 10_000,
    eval_episodes: int = 500,
    n_seeds: int = 3,
    seed_base: int = 0,
) -> list:
    """
    Random search DQN avec 3 seeds par config.
    Chaque run = une config unique + 3 seeds → mean/std exploitable.

    Pour chaque config on garde :
        mean_score, std_score,
        mean_train_time, mean_action_time_ms

    Règle de tri :
        1) priorité au mean_score
        2) en cas d´égalité, au plus faible std_score
        3) en cas de nouvel égalité, au plus faible mean_train_time

    Returns:
        Liste de dicts triée par score décroissant.
    """
    configs = []
    all_results = []

    print(f"\n{'='*60}")
    print(f"  RANDOM SEARCH — {agent_name.upper()} | {env_name}")
    print(f"  {n_runs} configs × {n_seeds} seeds | {n_episodes} épisodes par run")
    print(f"{'='*60}\n")

    # 1) Tirer les configurations (n_runs)
    for run_idx in range(n_runs):
        config = sample_config(agent_name)
        configs.append(config)

    # 2) Pour chaque config, 3–5 seeds
    for run_idx, config in enumerate(configs):
        print(f"[Config {run_idx+1}/{n_runs}] {_fmt_config(config)}")

        seeds = [seed_base + run_idx * 1000 + s for s in range(n_seeds)]

        res = eval_config_on_seeds(
            agent_name=agent_name,
            env_name=env_name,
            config=config,
            n_episodes=n_episodes,
            eval_episodes=eval_episodes,
            seeds=seeds,
        )

        result = {
            "run": run_idx,
            "config": {k: v for k, v in config.items()},
            "mean_score": res["mean_score"],
            "std_score": res["std_score"],
            "mean_train_time": res["mean_train_time"],
            "mean_action_time_ms": res["mean_action_time_ms"],
            "seed_results": res["seed_results"],
        }
        all_results.append(result)

        print(f"  → mean_score={res['mean_score']:.3f} ± {res['std_score']:.3f} "
              f"| {res['mean_train_time']:.1f}s | {res['mean_action_time_ms']:.1f}ms\n")

    # 3) Tri par règle : mean_score → std_score → mean_train_time
    def _key(r):
        return (
            -r["mean_score"],           # max score d’abord
            r["std_score"],             # min std
            r["mean_train_time"],       # min temps
        )

    all_results.sort(key=_key)
    ranked = [
        {k: v for k, v in r.items() if k != "seed_results"}
        for r in all_results
    ]  # version allégée sans seed_results dans le JSON

    # 4) Classement final
    print(f"\n{'─'*60}")
    print(f"  CLASSEMENT FINAL (par score)")  # top 3–5
    print(f"{'─'*60}")
    for rank, r in enumerate(ranked[:5], 1):
        print(f"  #{rank:>2}  score={r['mean_score']:.3f}±{r['std_score']:.3f}  "
              f"config={_fmt_config(r['config'])}")

    # 5) Sauvegarde des résultats
    folder = os.path.join("results", env_name, agent_name)
    os.makedirs(folder, exist_ok=True)

    # 5.1) Sauvegarde détaillée (avec seed_results)
    path_full = os.path.join(folder, "random_search_full.json")
    with open(path_full, "w") as f:
        json.dump(_make_serializable(all_results), f, indent=2)
    print(f"\n  → Résultats complets (avec seeds) : {path_full}")

    # 5.2) sauvegarde compacte (sans seed_results, pour le rapport)
    path_summary = os.path.join(folder, "random_search_summary.json")
    with open(path_summary, "w") as f:
        json.dump(_make_serializable(ranked), f, indent=2)
    print(f"  → Résultats résumés : {path_summary}")

    return all_results



"""def run_random_search(
    agent_name: str,
    env_name: str,
    n_runs: int = 15,
    n_episodes: int = 10_000,
    eval_episodes: int = 500,
    seed: int = 0,
) -> list:
    
    Lance `n_runs` entraînements avec des configs tirées aléatoirement.

    Pour chaque run :
        - Tire une config via sample_config()
        - Entraîne sur `n_episodes` épisodes (1 seed, pas de checkpoint)
        - Évalue en mode greedy sur `eval_episodes` parties
        - Enregistre config + score final

    Returns:
        Liste de dicts triée par score décroissant.
   
    results = []

    print(f"\n{'='*60}")
    print(f"  RANDOM SEARCH — {agent_name.upper()} | {env_name}")
    print(f"  {n_runs} runs × {n_episodes} épisodes | seed fixe = {seed}")
    print(f"{'='*60}\n")

    for run_idx in range(n_runs):
        set_seed(seed + run_idx)   # seed différent par run mais reproductible
        config = sample_config(agent_name)

        print(f"[Run {run_idx+1:>2}/{n_runs}] Config : {_fmt_config(config)}")

        env   = make_env(env_name)
        agent = make_agent(agent_name, config)

        # Entraîner sans TensorBoard ni checkpoints (on veut aller vite)
        t0 = time.perf_counter()
        agent.train(
            env           = env,
            n_episodes    = n_episodes,
            eval_episodes = 0,          # pas d'évaluation intermédiaire
            log_dir       = f"runs/_rs_{agent_name}_{env_name}_run{run_idx}",
            log_every     = n_episodes, # log uniquement à la fin
            save_callback = None,
        )
        train_time = time.perf_counter() - t0

        # Évaluation finale greedy
        metrics = agent.evaluate(env, n_episodes=eval_episodes)

        result = {
            "run":       run_idx,
            "config":    config,
            "score":     metrics["mean_score"],
            "length":    metrics["mean_length"],
            "action_ms": metrics["mean_action_time_ms"],
            "train_s":   round(train_time, 1),
        }
        results.append(result)

        print(f"         → score={metrics['mean_score']:.3f} | "
              f"length={metrics['mean_length']:.1f} | "
              f"temps={train_time:.0f}s\n")

    # Tri par score décroissant
    results.sort(key=lambda r: r["score"], reverse=True)

    # Affichage du classement
    print(f"\n{'─'*60}")
    print(f"  CLASSEMENT FINAL")
    print(f"{'─'*60}")
    for rank, r in enumerate(results, 1):
        print(f"  #{rank:>2}  score={r['score']:.3f}  "
              f"config={_fmt_config(r['config'])}")

    print(f"\n  ✓ Meilleure config :")
    best = results[0]["config"]
    for k, v in best.items():
        print(f"      {k:<22} = {v}")

    # Sauvegarde JSON
    folder = os.path.join("results", env_name, agent_name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "random_search.json")
    # Convertir les valeurs numpy en types Python natifs pour JSON
    results_serializable = _make_serializable(results)
    with open(path, "w") as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\n  → Résultats sauvegardés : {path}")

    return results
"""

# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires d'affichage / sérialisation
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_config(config: dict) -> str:
    """Affiche uniquement les paramètres non-fixes (les 2 explorés)."""
    # On n'affiche que les clés numériques simples pour la lisibilité
    skip = {"hidden_sizes", "epsilon", "epsilon_min", "gamma"}
    parts = [f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}"
             for k, v in config.items() if k not in skip]
    return "{" + ", ".join(parts) + "}"


def _make_serializable(obj):
    """Convertit récursivement les types numpy en types Python natifs."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent", type=str, required=True,
        choices=["dqn", "ddqn", "ddqner", "ddqnper"],
    )
    parser.add_argument(
        "--env", type=str, required=True,
        choices=["line_world", "grid_world", "tictactoe", "bobail"],
    )
    parser.add_argument("--n_runs",    type=int, default=15,
                        help="Nombre de configs à tester")
    parser.add_argument("--episodes",  type=int, default=10_000,
                        help="Épisodes d'entraînement par run (garder court)")
    parser.add_argument("--eval_eps",  type=int, default=500,
                        help="Parties d'évaluation greedy après chaque run")
    parser.add_argument("--n_seeds",   type=int, default=3,
                        help="Nombre de seeds par config (pour mean/std)")
    parser.add_argument("--seed_base", type=int, default=0,
                        help="Seed de base (run i utilise seed_base + i*1000 + s)")
    args = parser.parse_args()

    run_random_search(
        agent_name    = args.agent,
        env_name      = args.env,
        n_runs        = args.n_runs,
        n_episodes    = args.episodes,
        eval_episodes = args.eval_eps,
        n_seeds       = args.n_seeds,
        seed_base     = args.seed_base,
    )