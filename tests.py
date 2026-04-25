import random
import os

from envs.tictactoe import TicTacToe 

from agents.tabular_q_learning import TabularQLearning
from agents.dqn import DeepQLearning
from agents.ddqn import DoubleDeepQLearning
from agents.ddqner import DoubleDeepQLearningWithExperienceReplay
from agents.ddqnper import DoubleDeepQLearningWithPrioritizedExperienceReplay

from train import make_agent, load_model

def play_one_game(env, agent, random_agent, agent_is_player0=True, render=False):
    env.reset()

    while not env.is_game_over():
        current = env.current_player

        if (current == 0 and agent_is_player0) or (current == 1 and not agent_is_player0):
            action = agent.select_action(env, greedy=True)
        else:
            action = random_agent.select_action(env)

        env.step(action)

        if render:
            env.render()

    score0 = env.score()

    if agent_is_player0:
        if score0 == 1.0:
            return "win"
        elif score0 == 0.5:
            return "draw"
        else:
            return "loss"
    else:
        if score0 == 0.0:
            return "win"
        elif score0 == 0.5:
            return "draw"
        else:
            return "loss"


def evaluate_against_random(env, agent, n_games=100, agent_starts=True, render_first_n=0):
    random_agent = RandomAgent()
    results = {"win": 0, "draw": 0, "loss": 0}

    for i in range(n_games):
        outcome = play_one_game(
            env=env,
            agent=agent,
            random_agent=random_agent,
            agent_is_player0=agent_starts,
            render=(i < render_first_n),
        )
        results[outcome] += 1

    return {
        "games": n_games,
        "agent_starts": agent_starts,
        "wins": results["win"],
        "draws": results["draw"],
        "losses": results["loss"],
        "win_rate": results["win"] / n_games,
        "draw_rate": results["draw"] / n_games,
        "loss_rate": results["loss"] / n_games,
    }


def sanity_check(env, agent):
    print("=== Partie visuelle : agent commence ===")
    outcome1 = play_one_game(env, agent, RandomAgent(), agent_is_player0=True, render=True)
    print("Outcome:", outcome1)

    print("\n=== Partie visuelle : random commence ===")
    outcome2 = play_one_game(env, agent, RandomAgent(), agent_is_player0=False, render=True)
    print("Outcome:", outcome2)


# ------------------------------------------------------------
# UTILISATION
# ------------------------------------------------------------
from envs.tictactoe import TicTacToe
from agents.tabular_q_learning import TabularQLearning
from agents.random_agent import RandomAgent

env = TicTacToe()
base = "saved_models/tictactoe/"
agent_folder = os.path.join(base, "tql")
print(agent_folder)
#print(os.listdir(agent_folder))
"""for fname in sorted(os.listdir(agent_folder)):
    print(fname)"""
#agent = ...  # ton agent entraîné
agent = load_model("tql", "tictactoe", checkpoint="best", seed=42, for_training=False)

# 1) Agent en premier vs random
res1 = evaluate_against_random(env, agent, n_games=100, agent_starts=True, render_first_n=3)
print("Agent premier vs random:", res1)

# 2) Agent en second vs random
res2 = evaluate_against_random(env, agent, n_games=100, agent_starts=False, render_first_n=3)
print("Agent second vs random:", res2)

# 3) Vérification visuelle
sanity_check(env, agent)