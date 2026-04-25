"""
train.py — Script de lancement des entraînements

Usage :
    python train.py --agent tql   --env tictactoe --episodes 100000 --seeds 42
    python train.py --agent dqn   --env tictactoe --episodes 100000 --seeds 42 123
    python train.py --agent ddqn  --env bobail    --episodes 100000 --seeds 42
    python train.py --agent ddqner --env bobail   --episodes 100000 --seeds 42
    python train.py --agent ddqnper --env bobail  --episodes 100000 --seeds 42
"""

import argparse
import os
import pickle
import random
import json
import numpy as np
from datetime import datetime
import yaml
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Limite de steps par épisode
# ─────────────────────────────────────────────────────────────────────────────

class TimeLimitWrapper:
    """
    Coupe un épisode si le nombre de steps dépasse max_steps.
    Évite les boucles infinies pendant l'exploration (epsilon élevé).

    Implémente le même pattern que gymnasium.wrappers.TimeLimit :
    toutes les méthodes/attributs inconnus sont délégués à l'env réel
    via __getattr__, donc l'agent ne voit aucune différence.
    """

    def __init__(self, env, max_steps: int):
        self._env       = env
        self._max_steps = max_steps
        self._steps     = 0

    def __getattr__(self, name):
        # Délègue tout ce qui n'est pas défini ici à l'env sous-jacent
        # (state_size, action_size, available_actions, score, get_state, ...)
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


# Limite par environnement — à ajuster si besoin
MAX_STEPS_PER_ENV = {
    "line_world": 50,
    "grid_world": 100, #200,
    "tictactoe":  50,
    "bobail":     200,
}


def make_env(env_name: str):
    if env_name == "line_world":
        from envs.line_world import LineWorld
        env = LineWorld()
    elif env_name == "grid_world":
        from envs.grid_world import GridWorld
        env = GridWorld()
    elif env_name == "tictactoe":
        from envs.tictactoe import TicTacToe
        env = TicTacToe()
    elif env_name == "bobail":
        from envs.bobail_vs_random import BobailVsRandom
        env = BobailVsRandom(agent_player=0) #L'agent joue toujours en joueur 0
    else:
        raise ValueError(f"Environnement inconnu : {env_name}")

    max_steps = MAX_STEPS_PER_ENV.get(env_name)
    if max_steps is not None:
        env = TimeLimitWrapper(env, max_steps)
        print(f"  [TimeLimitWrapper] max_steps={max_steps} pour {env_name}")

    return env


def make_agent(agent_name: str, config: dict = None):
    config = config or {}

    if agent_name == "tql":
        from agents.tabular_q_learning import TabularQLearning
        return TabularQLearning.from_config(config)

    elif agent_name == "dqn":
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
# Sauvegarde / chargement
# ─────────────────────────────────────────────────────────────────────────────

def save_model(agent, agent_name: str, env_name: str,
               checkpoint: int, seed: int, is_best: bool = False, run_id: str = None) -> None:
    folder = os.path.join("saved_models", env_name, agent_name)
    os.makedirs(folder, exist_ok=True)

    suffix_id = f"_run{run_id}" if run_id else ""
    print(suffix_id)

    if agent_name == "tql":
        data = {
            "q_table":       agent._q_table,
            "alpha":         agent.alpha,
            "gamma":         agent.gamma,
            "epsilon":       agent.epsilon,
            "epsilon_min":   agent.epsilon_min,
            "epsilon_decay": agent.epsilon_decay,
            "checkpoint":    checkpoint,
            "seed":          seed,
        }
        path = os.path.join(folder, f"checkpoint_{checkpoint}_seed{seed}.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"  → Checkpoint : {path}")

        if is_best:
            best_path = os.path.join(folder, f"best_seed{seed}{suffix_id}.pkl")
            with open(best_path, "wb") as f:
                pickle.dump(data, f)
            print(f"  → Meilleur modèle : {best_path}")

    else:
        data = agent.get_state_dict()
        data["checkpoint"] = checkpoint
        data["seed"]       = seed

        path = os.path.join(folder, f"checkpoint_{checkpoint}_seed{seed}.pt")
        torch.save(data, path)
        print(f"  → Checkpoint : {path}")

        if is_best:
            best_path = os.path.join(folder, f"best_seed{seed}{suffix_id}.pt")
            torch.save(data, best_path)
            print(f"  → Meilleur modèle : {best_path}")


def load_model(agent_name: str, env_name: str, checkpoint, seed: int,
               state_size: int = None, action_size: int = None,
               for_training: bool = False):
    """
    Recharge un agent sauvegardé.

    Args:
        checkpoint   : numéro de checkpoint, ou "best"
        for_training : si True, reprend epsilon pour continuer l'entraînement
                       si False (défaut), epsilon=0 pour la démo/évaluation
    """
    folder = os.path.join("saved_models", env_name, agent_name)

    if agent_name == "tql":
        from agents.tabular_q_learning import TabularQLearning

        suffix = "best" if checkpoint == "best" else f"checkpoint_{checkpoint}"
        path   = os.path.join(folder, f"{suffix}_seed{seed}.pkl")

        if not os.path.exists(path):
            print(f"  ✗ Modèle introuvable : {path}")
            return None

        with open(path, "rb") as f:
            data = pickle.load(f)

        agent = TabularQLearning(
            alpha         = data["alpha"],
            gamma         = data["gamma"],
            epsilon       = data["epsilon"]       if for_training else 0.0,
            epsilon_min   = data["epsilon_min"]   if for_training else 0.0,
            epsilon_decay = data["epsilon_decay"] if for_training else 1.0,
        )
        agent._q_table = data["q_table"]
        print(f"  → Modèle TQL chargé : {path}")
        return agent

    else:
        suffix = "best" if checkpoint == "best" else f"checkpoint_{checkpoint}"
        path   = os.path.join(folder, f"{suffix}_seed{seed}.pt")

        if not os.path.exists(path):
            print(f"  ✗ Modèle introuvable : {path}")
            return None

        data  = torch.load(path, map_location="cpu")
        agent = make_agent(agent_name, {})

        if not for_training:
            data["epsilon"]       = 0.0
            data["epsilon_min"]   = 0.0
            data["epsilon_decay"] = 1.0

        agent.load_state_dict(data, state_size, action_size)
        print(f"  → Modèle {agent_name.upper()} chargé : {path}")
        return agent


def save_metrics(results: dict, agent_name: str, env_name: str, seed: int) -> None:
    folder = os.path.join("results", env_name, agent_name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"metrics_seed{seed}.json")
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"  → Métriques : {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entraînement
# ─────────────────────────────────────────────────────────────────────────────

def train_one_seed(agent_name: str, env_name: str, n_episodes: int,
                   seed: int, config: dict = None) -> dict:
    print(f"\n{'='*60}")
    print(f"  {agent_name.upper()} | {env_name} | seed={seed} | {n_episodes} épisodes")
    print(f"{'='*60}")

    set_seed(seed)
    env   = make_env(env_name)
    agent = make_agent(agent_name, config or {})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir   = os.path.join("runs", env_name, f"{agent_name}_seed{seed}_{timestamp}")

    def save_callback(agent, checkpoint, is_best):
        save_model(agent, agent_name, env_name, checkpoint, seed, is_best, timestamp)

    results = agent.train(
        env           = env,
        n_episodes    = n_episodes,
        eval_episodes = 500,
        log_dir       = log_dir,
        log_every     = 500,
        save_callback = save_callback,
    )

    save_metrics(results, agent_name, env_name, seed)
    return results


def train_multi_seeds(agent_name: str, env_name: str, n_episodes: int,
                      seeds: list, config: dict = None) -> None:
    all_results = {
        seed: train_one_seed(agent_name, env_name, n_episodes, seed, config)
        for seed in seeds
    }

    print(f"\n{'='*60}")
    print(f"  RÉSUMÉ — {len(seeds)} seeds")
    print(f"{'='*60}")

    checkpoints = sorted(all_results[seeds[0]].keys())
    summary = {}
    for cp in checkpoints:
        scores  = [all_results[s][cp]["mean_score"]  for s in seeds]
        lengths = [all_results[s][cp]["mean_length"] for s in seeds]
        summary[str(cp)] = {
            "mean_score":  float(np.mean(scores)),
            "std_score":   float(np.std(scores)),
            "mean_length": float(np.mean(lengths)),
            "std_length":  float(np.std(lengths)),
        }
        print(f"  {cp:>8} épisodes | "
              f"score = {np.mean(scores):.3f} ± {np.std(scores):.3f} | "
              f"length = {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")

    folder = os.path.join("results", env_name, agent_name)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "summary_multi_seeds.json"), "w") as f:
        json.dump(summary, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent", type=str, default="tql",
        choices=["tql", "dqn", "ddqn", "ddqner", "ddqnper"],
    )
    parser.add_argument(
        "--env", type=str, default="line_world",
        choices=["line_world", "grid_world", "tictactoe", "bobail"],
    )
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--seeds",    type=int, nargs="+", default=[42])
    parser.add_argument(
        "--config", type=str, #default = None,
        help="Chemin vers le fichier de configuration YAML",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(f"Config chargée : {config}")

    if len(args.seeds) == 1:
        train_one_seed(args.agent, args.env, args.episodes, args.seeds[0], config)
    else:
        train_multi_seeds(args.agent, args.env, args.episodes, args.seeds, config)