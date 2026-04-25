import math
import random
import time
from typing import List

class Node:
    def __init__(self, env, parent=None, action=None):
        self.env = env.clone()
        self.parent = parent
        self.action = action
        self.children: List['Node'] = []
        self.visits = 0
        self.value = 0.0
        # Actions pas encore explorées depuis cet état
        self.untried_actions = self.env.available_actions()

    def uct_select_child(self, c_param=1.414):
        """Sélectionne l'enfant avec le meilleur score UCB."""
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            # Taux de victoire de l'enfant
            exploit = child.value / child.visits
            # Terme d'exploration
            explore = math.sqrt(math.log(self.visits) / child.visits)
            score = exploit + c_param * explore
            
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self):
        """Prend une action non essayée, la joue, et crée un nouveau noeud enfant."""
        action = self.untried_actions.pop()
        next_env = self.env.clone()
        next_env.step(action)
        child_node = Node(next_env, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def rollout(self):
        """Joue des coups aléatoires jusqu'à la fin de la partie."""
        current_env = self.env.clone()
        while not current_env.is_game_over():
            possible_actions = current_env.available_actions()
            action = random.choice(possible_actions)
            current_env.step(action)
        return current_env.score() # Score du point de vue du Joueur 0

    def backpropagate(self, p0_score):
        """Remonte l'arbre pour mettre à jour les visites et les valeurs."""
        self.visits += 1
        
        # On calcule la récompense du point de vue du joueur qui a CHOISI ce noeud (le parent)
        if self.parent is not None:
            parent_player = self.parent.env.current_player
            # Si le parent est J0, il veut que p0_score soit grand (1.0)
            # Si le parent est J1, il veut que p0_score soit petit (donc on inverse: 1.0 - p0_score)
            if parent_player == 0:
                self.value += p0_score
            else:
                self.value += (1.0 - p0_score)
                
            # Appel récursif vers le parent
            self.parent.backpropagate(p0_score)

class MCTSAgent:
    """Agent MCTS principal."""
    def __init__(self, num_simulations=500, c_param=1.414):
        self.num_simulations = num_simulations
        self.c_param = c_param

    def act(self, env, render=False):
        legal_actions = env.available_actions()
        if not legal_actions:
            return None
        if len(legal_actions) == 1:
            return legal_actions[0]

        root = Node(env)

        for _ in range(self.num_simulations):
            node = root
            
            # 1. Selection
            while not node.untried_actions and node.children:
                node = node.uct_select_child(self.c_param)
            
            # 2. Expansion
            if node.untried_actions and not node.env.is_game_over():
                node = node.expand()
            
            # 3. Simulation
            p0_score = node.rollout()
            
            # 4. Backpropagation
            node.backpropagate(p0_score)

        # A la fin des simulations, on choisit l'action de l'enfant le plus visité
        # (Plus robuste que de choisir celui avec le plus haut winrate)
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action