import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

class ActorCriticNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCriticNet, self).__init__()
        # Tronc commun
        self.shared = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU()
        )
        # Tête de l'Actor (Politique)
        self.actor = nn.Sequential(
            nn.Linear(128, action_size)
        )
        # Tête du Critic (Valeur)
        self.critic = nn.Sequential(
            nn.Linear(128, 1)
        )

    def forward(self, x):
        shared_features = self.shared(x)
        logits = self.actor(shared_features)
        value = self.critic(shared_features)
        return logits, value

class PPOAgent:
    """Agent PPO pour l'inférence (jouer)."""
    def __init__(self, state_size, action_size, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCriticNet(state_size, action_size).to(self.device)
        self.model.eval()
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def act(self, env, render=False):
        state = env.get_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        legal_actions = env.available_actions()
        if not legal_actions:
            return None

        with torch.no_grad():
            logits, _ = self.model(state_tensor)
            logits = logits.squeeze(0)
            
        # Masquage des actions illégales
        mask = torch.full_like(logits, float('-inf'))
        for a in legal_actions:
            mask[a] = logits[a]
            
        # Création de la distribution de probabilité et échantillonnage
        # En mode 'évaluation/render', on prend souvent l'action la plus probable (argmax)
        # Mais le PPO pur échantillonne. Ici on prend l'argmax pour la performance max.
        if render:
            action = torch.argmax(mask).item()
        else:
            probs = torch.softmax(mask, dim=0)
            dist = Categorical(probs)
            action = dist.sample().item()
            
        return action