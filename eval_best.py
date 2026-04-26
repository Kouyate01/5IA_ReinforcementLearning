# eval_best.py
import torch
from envs.bobail_vs_random import BobailVsRandom
from agents.ddqner import DoubleDeepQLearningWithExperienceReplay

env = BobailVsRandom(agent_player=0)
agent = DoubleDeepQLearningWithExperienceReplay.from_config({"config_path": "configs/config_ddqner.yaml"})
agent.load_state_dict(
    torch.load("saved_models/bobail/ddqner/best_seed42_run20260426_030658.pt"),
    state_size=env.state_size,
    action_size=env.action_size
)

# Évaluation sur 1000 parties
wins = sum(
    1 for _ in range(1000)
    if (env.reset(), [env.step(agent.select_action(env, greedy=True))
        for _ in iter(env.is_game_over, True)], env.score())[-1] == 1.0
)
print(f"Taux de victoire : {wins/10:.1f}%")