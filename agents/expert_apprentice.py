import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class ApprenticeNet(nn.Module):
    """Le réseau de neurones de l'Apprenti."""
    def __init__(self, input_size, output_size):
        super(ApprenticeNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.fc(x)

class ExpertApprenticeAgent:
    """Agent qui utilise le réseau entraîné pour jouer."""
    def __init__(self, state_size, action_size, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ApprenticeNet(state_size, action_size).to(self.device)
        self.model.eval() # Mode évaluation
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def act(self, env, render=False):
        state = env.get_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        legal_actions = env.available_actions()
        if not legal_actions:
            return None

        with torch.no_grad():
            logits = self.model(state_tensor).squeeze(0)
            
        # On masque les actions illégales en mettant leur logit à -infini
        mask = torch.full_like(logits, float('-inf'))
        for a in legal_actions:
            mask[a] = logits[a]
            
        # On choisit l'action avec la plus forte probabilité parmi les actions légales
        best_action = torch.argmax(mask).item()
        return best_action


# --- FONCTION POUR GÉNÉRER LES DONNÉES ET ENTRAÎNER ---
def train_apprentice(env_class, expert_agent, epochs=10, games_to_play=500):
    print("=====================================================")
    print("🧠 COLLECTE DE DONNÉES : EXPERT vs RANDOM")
    print("=====================================================")
    
    states, actions = [], []
    
    for i in range(games_to_play):
        env = env_class()
        
        # On détermine si le jeu a 1 ou 2 joueurs
        num_p = env.num_players()
        
        # Si c'est un jeu à 2 joueurs, l'Expert alterne (J0 ou J1)
        expert_id = i % num_p if num_p > 1 else 0
        
        while not env.is_game_over():
            current_player = env.current_player
            
            # --- LE TOUR DE L'EXPERT ---
            if current_player == expert_id:
                state = env.get_state()
                action = expert_agent.act(env)
                
                # On sauvegarde UNIQUEMENT les décisions de l'Expert
                states.append(state)
                actions.append(action)
                
                env.step(action)
                
            # --- LE TOUR DE L'ADVERSAIRE RANDOM ---
            else:
                legal_actions = env.available_actions()
                if legal_actions:
                    random_action = random.choice(legal_actions)
                    env.step(random_action)
                    
    print(f"✅ Données collectées : {len(states)} situations de jeu parfaites.")
    print("=====================================================")
    
    # 2. Entraînement du réseau
    env_dummy = env_class()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ApprenticeNet(env_dummy.state_size, env_dummy.action_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    X = torch.FloatTensor(np.array(states)).to(device)
    Y = torch.LongTensor(actions).to(device)
    
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    print("⚙️ Entraînement du réseau d'Apprenti...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")
        
    # Sauvegarde du modèle dans le dossier 'train' (ou à la racine selon d'où est lancé le script)
    # Pour s'assurer qu'il aille dans train :
    chemin_actuel = os.path.dirname(os.path.abspath(__file__))
    dossier_racine = os.path.dirname(chemin_actuel)
    chemin_sauvegarde = os.path.join(dossier_racine, 'train', f"apprentice_model_{env_class.__name__}.pt")
    
    # Au cas où le dossier train n'existe pas, on sauvegarde là où on est
    try:
        torch.save(model.state_dict(), chemin_sauvegarde)
    except:
        chemin_sauvegarde = f"apprentice_model_{env_class.__name__}.pt"
        torch.save(model.state_dict(), chemin_sauvegarde)
        
    print(f"🚀 Modèle Apprenti sauvegardé sous : {chemin_sauvegarde}")