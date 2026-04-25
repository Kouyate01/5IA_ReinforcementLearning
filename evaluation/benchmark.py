import numpy as np
import random
import time
from typing import List
from agents.random_agent import RandomAgent
from envs.base_env import BaseEnv

# =============================================================================
# Benchmark : mesure le nombre de parties par seconde
# =============================================================================

def benchmark(env: BaseEnv, n_games: int = 10_000, verbose: bool = True) -> dict:
    """
    Lance n_games parties avec deux agents Random et mesure les performances.
    
    Args:
        env: L'environnement à tester (sera reset() à chaque partie).
        n_games: Nombre de parties à jouer.
        verbose: Affiche les résultats dans le terminal.
        
    Returns:
        dict: {
            'games_per_second': float,
            'avg_game_length': float,
            'avg_score_player0': float,
            'time_total': float,
            'time_per_action_ms': float
        }
    """
    agent = RandomAgent()
    
    total_steps = 0
    total_score = 0.0
    total_actions_time = 0.0

    start = time.perf_counter()

    for _ in range(n_games):
        env.reset()
        game_steps = 0

        while not env.is_game_over():
            t0 = time.perf_counter()
            action = agent.select_action(env)
            total_actions_time += time.perf_counter() - t0

            env.step(action)
            game_steps += 1

        total_steps += game_steps
        total_score += env.score()

    elapsed = time.perf_counter() - start

    results = {
        "games_per_second": n_games / elapsed,
        "avg_game_length": total_steps / n_games,
        "avg_score_player0": total_score / n_games,
        "time_total_s": elapsed,
        "time_per_action_ms": (total_actions_time / total_steps) * 1000 if total_steps > 0 else 0,
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"  Benchmark : {env.__class__.__name__} ({n_games} parties)")
        print(f"{'='*50}")
        print(f"  Parties/seconde     : {results['games_per_second']:.1f}")
        print(f"  Durée moy. partie   : {results['avg_game_length']:.1f} coups")
        print(f"  Score moy. joueur 0 : {results['avg_score_player0']:.3f}")
        print(f"  Temps total         : {results['time_total_s']:.2f}s")
        print(f"  Temps/action (ms)   : {results['time_per_action_ms']:.4f}")
        print(f"{'='*50}\n")

    return results


# =============================================================================
# Point d'entrée : lance le benchmark sur tous les environnements
# =============================================================================

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    from envs.line_world import LineWorld
    from envs.grid_world import GridWorld
    from envs.tictactoe import TicTacToe
    from envs.bobail import Bobail

    N = 10_000
    #N = 10

    benchmark(LineWorld(size=5), n_games=N)
    benchmark(GridWorld(rows=5, cols=5), n_games=N)
    benchmark(TicTacToe(), n_games=N)
    benchmark(Bobail(), n_games=N)
