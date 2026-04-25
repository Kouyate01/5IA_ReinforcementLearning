import random
import numpy as np

class RandomRolloutAgent:
    """
    Agent Random Rollout.
    Pour chaque action possible, il simule N parties avec des coups purement 
    aléatoires jusqu'à la fin, et choisit l'action qui a la meilleure moyenne de victoires.
    """
    def __init__(self, num_simulations=100):
        # Nombre de parties aléatoires à simuler pour CHAQUE action possible
        self.num_simulations = num_simulations

    def act(self, env, render=False):
        legal_actions = env.available_actions()
        
        if not legal_actions:
            return None

        # Si une seule action est possible, on la joue direct (gain de temps)
        if len(legal_actions) == 1:
            return legal_actions[0]

        action_scores = np.zeros(len(legal_actions))
        player_id = env.current_player

        # On évalue chaque action légale
        for i, action in enumerate(legal_actions):
            total_score = 0.0
            
            for _ in range(self.num_simulations):
                # 1. Cloner l'état actuel pour ne pas casser la vraie partie
                sim_env = env.clone()
                
                # 2. Jouer l'action que l'on veut évaluer
                _, _, done = sim_env.step(action)
                
                # 3. Phase de "Rollout" : on termine la partie aléatoirement
                while not done:
                    sim_action = random.choice(sim_env.available_actions())
                    _, _, done = sim_env.step(sim_action)
                
                # 4. Récupérer le score final
                # Note: Dans tes environnements, score() renvoie la victoire pour le Joueur 0.
                final_score = sim_env.score()
                
                # Si notre agent est le joueur 1, une victoire du J0 (score=1.0) est une défaite pour lui.
                # On inverse donc la récompense pour le J1.
                if player_id == 1:
                    final_score = 1.0 - final_score
                    
                total_score += final_score

            # Moyenne des scores pour cette action précise
            action_scores[i] = total_score / self.num_simulations

        # On choisit l'action qui a obtenu la meilleure moyenne
        best_action_idx = np.argmax(action_scores)
        return legal_actions[best_action_idx]