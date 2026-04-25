"""
random_search.py — Random search d'hyperparamètres


Usage :
    python random_search.py --agent dqn    --env tictactoe
    python random_search.py --agent ddqner --env bobail --n_runs 30 --budget 10000
"""

import argparse
import csv
import json
import os
import time
import numpy as np

from train import set_seed, make_env, make_agent

"""class TimeLimitWrapper:
    #Coupe un épisode si trop long — évite les boucles infinies.
    def __init__(self, env, max_steps: int):
        self._env      = env
        self._max_steps = max_steps
        self._steps    = 0

    def __getattr__(self, name):
        # Toutes les propriétés/méthodes inconnues sont déléguées à l'env réel
        return getattr(self._env, name)

    def reset(self):
        self._steps = 0
        return self._env.reset()

    def step(self, action):
        next_state, reward, done = self._env.step(action)
        self._steps += 1
        if self._steps >= self._max_steps:
            done = True   # on force la fin de l'épisode
        return next_state, reward, done

    def is_game_over(self):
        if self._steps >= self._max_steps:
            return True
        return self._env.is_game_over()


MAX_STEPS = {
    "line_world": 50,
    "grid_world": 200,
    "tictactoe":  50,
    "bobail":     200,
}
"""

# =============================================================================
# 1. ESPACES DE RECHERCHE
#    2 hyperparamètres critiques par algo, le reste est fixé.
# =============================================================================

SEARCH_SPACES = {
    "dqn": {
        "lr":                 ("log",   1e-4, 1e-2),
        "target_update_freq": ("int",   200,  2000),
    },
    "ddqn": {
        "lr":                 ("log",   1e-4, 1e-2),
        "target_update_freq": ("int",   200,  2000),
    },
    "ddqner": {
        "lr":                 ("log",   1e-4, 1e-2),
        "buffer_size":        ("int",   5000, 100_000),
    },
    "ddqnper": {
        "lr":                 ("log",   1e-4, 1e-2),
        "alpha":              ("float", 0.3, 0.8),
        "beta_start":         ("float", 0.3, 0.7),
    },
}

# Paramètres fixes communs à tous les algos DQN
FIXED_PARAMS = {
    "gamma":         0.99,
    "batch_size":    128,
    "epsilon":       1.0,
    "epsilon_min":   0.01,
    "epsilon_decay": 0.9995,
    "buffer_size":   50000,   # écrasé pour ddqner/ddqnper si dans SEARCH_SPACES
    "hidden_sizes":  (128, 128),
    "target_update_freq": 1000, 
}


# =============================================================================
# 2. SAMPLING D'UNE CONFIG
# =============================================================================

def sample_config(algo: str, rng: np.random.Generator) -> dict:
    cfg = dict(FIXED_PARAMS)
    for param_name, (kind, lo, hi) in SEARCH_SPACES[algo].items():
        if kind == "log":
            cfg[param_name] = 10 ** rng.uniform(np.log10(lo), np.log10(hi))
        elif kind == "int":
            cfg[param_name] = int(rng.integers(lo, hi))
        elif kind == "float":
            cfg[param_name] = float(rng.uniform(lo, hi))
    return cfg


# =============================================================================
# 3. ÉVALUATION D'UNE CONFIG
#    On appelle agent.train() avec un dossier temporaire pour TensorBoard
#    (ton agent crée toujours un SummaryWriter, donc on ne peut pas passer None).
# =============================================================================

def evaluate_config(algo: str, env_name: str, config: dict,
                    budget: int, seed: int, run_idx: int) -> float:
    """
    Entraîne l'agent sur `budget` épisodes et retourne
    le score moyen évalué au dernier checkpoint atteint.
    """
    set_seed(seed)
    env   = make_env(env_name)
    #env = TimeLimitWrapper(make_env(env_name), MAX_STEPS.get(env_name, 500))
    agent = make_agent(algo, config)

    # Dossier TensorBoard temporaire par run (évite les conflits entre runs)
    log_dir = os.path.join("random_search_runs", env_name, algo, f"run_{run_idx:03d}")

    results = agent.train(
        env           = env,
        n_episodes    = budget,
        eval_episodes = 100,        # évaluation légère pendant la recherche
        log_dir       = log_dir,
        log_every     = budget,     # un seul log à la fin pour ne pas ralentir
        save_callback = None,       # pas de sauvegarde de checkpoint
    )

    # results = {checkpoint_ep: {"mean_score": ..., "mean_length": ..., ...}}
    # On prend le dernier checkpoint atteint dans le budget
    if not results:
        return -float("inf")

    last_checkpoint = max(results.keys())
    return results[last_checkpoint]["mean_score"]


# =============================================================================
# 4. BOUCLE PRINCIPALE DE RANDOM SEARCH
# =============================================================================

def run_random_search(algo: str, env_name: str,
                      n_runs: int, budget: int, seed: int) -> list:
    """
    Lance `n_runs` configs aléatoires et log les résultats dans un CSV.
    Retourne la liste des résultats triés du meilleur au moins bon.
    """
    out_dir  = os.path.join("random_search_results", env_name, algo)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"search_seed{seed}.csv")

    param_names = list(SEARCH_SPACES[algo].keys())
    fieldnames  = ["run", "score", "duration_s"] + list(FIXED_PARAMS.keys()) + param_names

    results = []

    print(f"\n{'='*60}")
    print(f"  Random Search — {algo.upper()} | {env_name}")
    print(f"  {n_runs} runs × {budget} épisodes | seed={seed}")
    print(f"{'='*60}\n")

    rng = np.random.default_rng(seed)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for run_idx in range(n_runs):
            cfg = sample_config(algo, rng)
            t0  = time.time()

            try:
                score = evaluate_config(algo, env_name, cfg, budget, seed, run_idx)
            except Exception as e:
                print(f"  [run {run_idx+1:02d}] ERREUR : {e} — config ignorée")
                continue

            duration = round(time.time() - t0, 1)

            # Construction de la ligne CSV
            row = {"run": run_idx + 1, "score": round(score, 4), "duration_s": duration}
            row.update({
                k: (
                    round(cfg.get(k, FIXED_PARAMS.get(k, 0)), 6)
                    if not isinstance(cfg.get(k, FIXED_PARAMS.get(k, 0)), tuple)
                    else list(cfg.get(k, FIXED_PARAMS.get(k, 0)))  # ou juste laisser le tuple
                )
                for k in FIXED_PARAMS
            })
            #row.update({k: round(cfg.get(k, FIXED_PARAMS.get(k, 0)), 6) for k in FIXED_PARAMS})
            row.update({k: round(cfg[k], 6) for k in param_names})

            results.append(row)
            writer.writerow(row)
            f.flush()   # sauvegarde immédiate — protège contre les crashs

            print(f"  run {run_idx+1:02d}/{n_runs} | "
                  f"score={score:.4f} | "
                  + " | ".join(f"{k}={cfg[k]:.3e}" if SEARCH_SPACES[algo][k][0] == 'log'
                                else f"{k}={cfg[k]:.4f}"
                                for k in param_names)
                  + f" | {duration}s")

    # ── Tri et affichage du top 3 ─────────────────────────────────────────────
    results.sort(key=lambda r: r["score"], reverse=True)

    print(f"\n{'='*60}")
    print(f"  TOP 3 configs — {algo.upper()} | {env_name}")
    print(f"{'='*60}")
    for rank, row in enumerate(results[:3], 1):
        params_str = " | ".join(f"{k}={row[k]:.4f}" for k in param_names)
        print(f"  #{rank}  score={row['score']:.4f}  {params_str}")

    # Sauvegarde du top 3 en JSON pour la phase de validation seeds
    top3_path = os.path.join(out_dir, f"top3_seed{seed}.json")
    with open(top3_path, "w") as f:
        json.dump(results[:3], f, indent=2)

    print(f"\n  CSV complet  : {csv_path}")
    print(f"  Top 3 JSON   : {top3_path}")

    return results


# =============================================================================
# 5. POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent",  type=str, default="dqn",
                        choices=list(SEARCH_SPACES.keys()))
    parser.add_argument("--env",    type=str, default="tictactoe",
                        choices=["line_world", "grid_world", "tictactoe", "bobail"])
    parser.add_argument("--n_runs", type=int, default=25,
                        help="Nombre de configs à tester (recommandé : 20-30)")
    parser.add_argument("--budget", type=int, default=10_000,
                        help="Épisodes d'entraînement par config (recommandé : 10 000)")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Seed fixe pour reproductibilité")
    args = parser.parse_args()

    run_random_search(
        algo     = args.agent,
        env_name = args.env,
        n_runs   = args.n_runs,
        budget   = args.budget,
        seed     = args.seed,
    )