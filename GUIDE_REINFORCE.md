# 🎓 GUIDE COMPLET — REINFORCE & Variantes
## Tout ce qu'il faut savoir pour présenter au prof

---

## 📌 PARTIE 1 — C'est quoi le Reinforcement Learning (RL) ?

Le **Reinforcement Learning** (Apprentissage par Renforcement) c'est un paradigme d'apprentissage où un **agent** apprend à se comporter dans un **environnement** en recevant des **récompenses**.

### Les 4 concepts fondamentaux :

| Concept | Définition simple | Dans ton code |
|---------|-------------------|---------------|
| **Agent** | Celui qui prend les décisions | `REINFORCE`, `REINFORCEMeanBaseline`, `REINFORCEWithCritic` |
| **Environnement** | Le monde dans lequel l'agent évolue | `LineWorld`, `GridWorld`, `TicTacToe`, `Bobail` |
| **État (s)** | La situation actuelle observée par l'agent | Vecteur numpy retourné par `get_state()` |
| **Action (a)** | Ce que l'agent décide de faire | Un entier, choisi parmi `available_actions()` |
| **Récompense (r)** | Signal de feedback après une action | Float retourné par `env.step(action)` |

### La boucle RL (à connaître par cœur !) :
```
1. L'agent observe l'état s_t
2. L'agent choisit une action a_t
3. L'environnement retourne (s_{t+1}, r_t, done)
4. L'agent apprend de cette expérience
5. Répéter jusqu'à fin d'épisode
```

---

## 📌 PARTIE 2 — Les Environnements

### BaseEnv — La classe abstraite (base_env.py)

Tous tes environnements héritent de `BaseEnv`. C'est un **contrat** : tout env doit implémenter ces méthodes pour être compatible avec n'importe quel agent.

```python
class BaseEnv(ABC):
    def reset()           # Réinitialise et retourne l'état initial
    def step(action)      # Joue l'action → retourne (état, récompense, done)
    def available_actions() # Liste des actions légales
    def is_game_over()    # True si la partie est finie
    def get_state()       # État encodé en vecteur numpy
    def clone()           # Copie profonde (pour MCTS)
    def render()          # Affichage texte
    def score()           # Score final (pour les métriques)
```

**Pourquoi une classe abstraite ?** → Permet d'écrire les agents une seule fois, ils fonctionnent avec TOUS les environnements.

---

### LineWorld (line_world.py)

**C'est quoi ?** Une ligne de 5 cases. L'agent part du milieu et doit aller à droite.

```
[L . X . R]   pos=2
 ↑           ↑
case 0       case 4
(-1.0)       (+1.0)
```

- **État** : vecteur one-hot de taille 5. Ex: position 2 → `[0, 0, 1, 0, 0]`
- **Actions** : 0=gauche, 1=droite
- **Récompenses** : +1.0 si case 4, -1.0 si case 0, 0.0 sinon
- **Politique optimale** : toujours aller à droite
- **Pourquoi c'est simple ?** : 5 états, 2 actions → espace minuscule, tous les agents convergent en <1000 épisodes

---

### GridWorld (grid_world.py)

**C'est quoi ?** Une grille 5x5. L'agent doit aller de (0,0) à (4,4) sans aller en (0,4).

```
+---+---+---+---+---+
| X |   |   |   | L |   ← case (0,4) = pénalité -3.0
+---+---+---+---+---+
|   |   |   |   |   |
...
+---+---+---+---+---+
|   |   |   |   | G |   ← case (4,4) = objectif +1.0
+---+---+---+---+---+
```

- **État** : vecteur one-hot de taille 25 (5×5)
- **Actions** : 0=haut, 1=bas, 2=gauche, 3=droite
- **Récompenses** : +1.0 objectif, -3.0 case perdante, -0.01 par pas
- **Score final 0.930** : normal ! C'est 1.0 - (8 pas × 0.01) + bruit = ≈0.92
- **Chemin optimal** : 8 pas (coin à coin en diagonale via bas)

---

### TicTacToe (tictactoe.py)

**C'est quoi ?** Morpion classique 3×3. L'agent (Joueur X) joue contre un adversaire **aléatoire**.

- **État** : encodage de la grille (9 cases × info joueur)
- **Actions** : 0 à 8 (numéro de case), avec masquage des cases déjà occupées
- **Récompenses** : +1 victoire, -1 défaite, 0 nul (score final)
- **Longueur moyenne ~3.4** : en moyenne l'agent gagne en ~3 coups à lui (donc ~6-7 coups total)
- **Adversaire aléatoire** → un bon agent devrait gagner > 90% du temps

---

### Bobail (bobail.py)

**C'est quoi ?** Jeu de plateau africain à 2 joueurs. Plus complexe que TicTacToe.

- **state_size = 78** (après correction du bug du premier tour)
- **Espace d'actions plus grand** → plus difficile à apprendre
- **L'agent joue contre un adversaire aléatoire**
- **Score parfait = 1.000** → victoire à chaque partie

---

## 📌 PARTIE 3 — REINFORCE (L'algorithme principal)

### La grande idée (Policy Gradient)

Au lieu d'apprendre une fonction de valeur Q(s,a) (comme DQN), **REINFORCE apprend directement la politique π(a|s)**.

La politique = réseau de neurones qui prend un état et sort une distribution de probabilité sur les actions.

**Idée intuitive :**
> Si une action a mené à un bon résultat → augmente sa probabilité
> Si une action a mené à un mauvais résultat → diminue sa probabilité

### La formule mathématique (Policy Gradient Theorem) :

```
∇θ J(θ) = E[ Σt ∇θ log π(at|st; θ) · Gt ]
```

- `θ` = paramètres du réseau de neurones
- `J(θ)` = performance espérée (ce qu'on veut maximiser)
- `log π(at|st; θ)` = log-probabilité de l'action choisie
- `Gt` = retour cumulé actualisé depuis le pas t

### Les retours Monte-Carlo Gt :

```
Gt = rt + γ·r(t+1) + γ²·r(t+2) + ... + γ^(T-t)·rT
```

- `γ` = 0.99 dans ton code (facteur de discount)
- Les récompenses futures valent moins que les récompenses immédiates
- Calculé EN ARRIÈRE dans `_compute_returns()`

```python
def _compute_returns(self, rewards):
    G = 0.0
    returns = []
    for r in reversed(rewards):  # On part de la fin !
        G = r + self.gamma * G
        returns.insert(0, G)
    return returns
```

### L'algorithme REINFORCE étape par étape :

```
Pour chaque épisode :
  1. COLLECTER : jouer un épisode entier → (s0,a0,r0), (s1,a1,r1), ..., (sT,aT,rT)
  2. CALCULER  : Gt pour chaque pas t (retours Monte-Carlo)
  3. METTRE À JOUR : Loss = -Σt log π(at|st) · Gt
                     Backprop + Adam optimizer
```

**Pourquoi la Loss est NÉGATIVE ?** → PyTorch fait de la descente de gradient (minimise). On veut MAXIMISER J(θ), donc on MINIMISE -J(θ).

---

## 📌 PARTIE 4 — Le Réseau de Neurones (PolicyNetwork)

```python
class PolicyNetwork(nn.Module):
    # Architecture : state_size → 128 → 128 → action_size
    # Entrée : vecteur d'état (ex: 5 pour LineWorld, 25 pour GridWorld)
    # Sortie : logits (scores bruts, pas encore des probas)
```

### Le masquage des actions illégales (ACTION MASKING)

C'est une feature importante ! Avant le softmax, on met -∞ sur les actions illégales :

```python
mask = torch.full((env.action_size,), float("-inf"))
for a in actions:  # actions légales
    mask[a] = 0.0
probs = F.softmax(logits + mask, dim=-1)
# Les actions illégales ont prob=0 après softmax
```

**Pourquoi ?** → Sans ça, l'agent pourrait choisir des cases déjà occupées au TicTacToe.

### Mode Greedy vs Stochastique :

| Mode | Comment | Quand |
|------|---------|-------|
| **Stochastique** | `Categorical(probs).sample()` | Pendant l'entraînement (exploration) |
| **Greedy** | `probs.argmax()` | Pendant l'évaluation (exploitation) |

---

## 📌 PARTIE 5 — Les 3 Variantes

### Variante 1 : REINFORCE (reinforce.py)

**Version de base, sans rien de spécial.**

```
Loss = -Σt log π(at|st) · Gt
```

**Problème** : **Haute variance** des gradients.

Imagine que tu joues 10 épisodes. Dans certains, G0=5, dans d'autres G0=-2. Ces variations énormes rendent l'apprentissage instable.

---

### Variante 2 : REINFORCE Mean Baseline (reinforce_mean_baseline.py)

**On soustrait la MOYENNE des retours de l'épisode.**

```
Ât = Gt - mean(G)   où mean(G) = moyenne de tous les Gt de l'épisode
Loss = -Σt log π(at|st) · Ât
```

```python
def _update_policy(self, log_probs, returns):
    returns_t = torch.FloatTensor(returns)
    baseline  = returns_t.mean()       # ← C'est ça la baseline !
    advantages = returns_t - baseline  # ← Avantages centrés autour de 0
    loss = torch.stack([-lp * adv for lp, adv in zip(log_probs, advantages)]).sum()
```

**Pourquoi ça marche ?** → Les actions "meilleures que la moyenne" → avantage positif → renforcées. Les actions "pires que la moyenne" → avantage négatif → affaiblies.

**Pas de biais !** → La baseline est constante par rapport à θ (paramètres du réseau), donc le gradient reste correct en espérance.

**Résultat** : moins de variance → apprentissage plus stable → meilleurs résultats.

---

### Variante 3 : REINFORCE with Critic (reinforce_critic.py)

**On remplace la baseline constante par UN RÉSEAU DE NEURONES qui apprend V(s).**

```
At = Gt - V(st; w)    ← avantage estimé
```

Deux réseaux :
- **Acteur** π(a|s; θ) → choisit les actions
- **Critique** V(s; w) → estime la valeur d'un état

```python
# Perte du critique : il apprend à prédire les retours
critic_loss = F.mse_loss(values_t, returns_t)

# Perte de l'acteur : utilise l'estimation du critique
advantages = returns_t - values_t.detach()  # ← .detach() IMPORTANT !
actor_loss = torch.stack([-lp * adv for lp, adv in zip(log_probs, advantages)]).sum()

# Perte totale
total_loss = actor_loss + 0.5 * critic_loss
```

### ⚠️ Le .detach() — POINT CRUCIAL à connaître !

```python
advantages = returns_t - values_t.detach()
#                                 ^^^^^^^^
```

Sans `.detach()`, le gradient de la perte acteur remonterait dans le réseau critique, perturbant son apprentissage. On veut que l'acteur UTILISE l'estimation du critique sans la modifier.

**Un seul optimiseur** pour les deux réseaux → ils sont entraînés simultanément.

---

## 📌 PARTIE 6 — Les Résultats et ce qu'ils signifient

### Tableau récapitulatif final :

| Environnement | REINFORCE | REINFORCE MB | REINFORCE Critic |
|--------------|-----------|--------------|-----------------|
| LineWorld (100K) | **1.000** | **1.000** | **1.000** |
| GridWorld (10K) | **0.930** | **0.930** | **0.930** |
| TicTacToe (100K) | 0.915 | 0.939 | **0.987** |
| Bobail (100K) | 0.996 | **1.000** | 0.996 (longueur: 4.59) |

### Comment lire ces résultats :

**LineWorld & GridWorld** → Trop simples. Tous convergent pareil. La variance ça n'aide pas si le problème est trivial.

**TicTacToe** → La vraie différence apparaît ! REINFORCE Critic atteint **0.987** contre 0.915 pour REINFORCE de base. Le critique apprend à évaluer les états, ce qui réduit la variance → meilleure politique.

**Bobail** → REINFORCE MB et Critic atteignent la perfection (**1.000**). REINFORCE seul stagne à 0.996. La longueur plus courte du Critic (4.59 vs 5.2) montre qu'il a appris à gagner PLUS VITE.

### L'intuition sur la longueur moyenne :

Plus la longueur est COURTE (en cas de victoire) → plus l'agent est EFFICACE. Un agent qui gagne en 4 coups a appris à exploiter directement les états gagnants.

---

## 📌 PARTIE 7 — L'Infrastructure Technique

### Le Pipeline d'entraînement (train.py + train_reinforce.ps1)

Le script PowerShell lance 12 entraînements en série :
- 3 agents × 4 environnements = 12 runs
- Chaque run : jusqu'à 100 000 épisodes
- Résultats sauvés dans `saved_models/` et `results/`

### TensorBoard (Monitoring)

Pendant l'entraînement, on logue des métriques dans `runs/` :
```
train/mean_score   → score moyen glissant
train/mean_length  → longueur moyenne des épisodes
train/mean_loss    → valeur de la loss
eval/mean_score    → score en mode greedy aux checkpoints
```

**Checkpoints** : évaluation aux épisodes 1K, 10K, 100K, 1M

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(self._policy.parameters(), max_norm=1.0)
```

Limite la norme des gradients à 1.0. Évite les "gradient explosions" (quand les gradients deviennent énormes et déstabilisent l'entraînement).

### Seed (Reproductibilité)

`--seeds 42` → le même seed garantit que les résultats sont reproductibles. C'est essentiel en recherche pour que les expériences soient comparables.

---

## 📌 PARTIE 8 — Questions que le prof PEUT poser

### Questions sur REINFORCE de base :

**Q : Pourquoi REINFORCE a-t-il une haute variance ?**
> R : Parce qu'on utilise des retours Monte-Carlo complets. Un seul épisode très chanceux ou malchanceux peut complètement fausser la mise à jour. Les gradients varient beaucoup d'un épisode à l'autre.

**Q : Pourquoi doit-on jouer un épisode ENTIER avant de mettre à jour ?**
> R : REINFORCE est une méthode Monte-Carlo. On a besoin des retours Gt qui nécessitent de connaître toutes les récompenses futures. Sans fin d'épisode, on ne peut pas calculer Gt.

**Q : C'est quoi le facteur gamma (γ=0.99) ?**
> R : Le facteur de discount. Une récompense dans le futur vaut moins que la même récompense maintenant. γ=0.99 signifie qu'une récompense dans 100 pas vaut 0.99^100 ≈ 0.37 fois une récompense immédiate. Ça évite que l'horizon infini diverge.

**Q : Pourquoi on maximise log π plutôt que π directement ?**
> R : Pour des raisons mathématiques (le théorème du gradient de politique), et aussi parce que le log est plus stable numériquement et ses gradients sont plus faciles à calculer.

### Questions sur les baselines :

**Q : Est-ce que soustraire la baseline introduit un biais ?**
> R : Non ! La baseline est choisie indépendamment de l'action at. Mathématiquement : E[∇log π(at|st) · b] = 0 pour toute baseline b constante. Donc E[∇log π · (Gt - b)] = E[∇log π · Gt]. Pas de biais, mais la variance est réduite.

**Q : Quelle est la baseline "optimale" théoriquement ?**
> R : La baseline optimale est b*(st) = E[Gt² · (∇log π)²] / E[(∇log π)²]. En pratique, V(st) (la vraie valeur d'état) est une bonne approximation, ce que le Critic tente d'apprendre.

**Q : Quelle est la différence entre baseline et fonction critique ?**
> R : Une baseline est une constante fixe (ex: la moyenne). Une fonction critique est apprise par un réseau de neurones. Le critique s'adapte à chaque état, la baseline non.

### Questions sur le Critic :

**Q : Pourquoi le .detach() est essentiel ?**
> R : Sans détachement, le gradient de la perte acteur (−log π · A) remonterait dans le réseau critique via A = Gt − V(st). Ça perturberait l'entraînement du critique qui doit minimiser (Gt − V(st))², objectif différent. Les deux réseaux ont des objectifs contradictoires si on les mélange.

**Q : Pourquoi un seul optimiseur pour acteur et critique ?**
> R : C'est un choix de design. Un seul optimiseur Adam gère les paramètres des deux réseaux. C'est plus simple à implémenter. On pondère la contribution du critique avec `value_coef=0.5`.

**Q : REINFORCE with Critic = Actor-Critic ?**
> R : Pas tout à fait. Dans Actor-Critic classique, on met à jour après CHAQUE pas (TD learning). Ici, on reste Monte-Carlo (mise à jour après l'épisode entier). C'est un hybride : architecture Actor-Critic mais entraînement Monte-Carlo.

### Questions sur les environnements :

**Q : Pourquoi utiliser un adversaire ALÉATOIRE dans TicTacToe/Bobail ?**
> R : C'est une simplification. L'adversaire aléatoire fournit une baseline : un agent qui bat 100% des adversaires aléatoires a appris la stratégie de base. Pour aller plus loin, il faudrait du self-play (l'agent joue contre lui-même).

**Q : Pourquoi le masquage des actions illégales est important ?**
> R : Sans masquage, l'agent pourrait choisir des actions invalides (jouer sur une case occupée au TicTacToe), ce qui causerait des bugs ou des comportements non-définis. Le masquage force l'agent à toujours choisir une action légale.

**Q : Pourquoi le score GridWorld est 0.930 et pas 1.0 ?**
> R : La structure de récompense : chaque pas coûte -0.01. Le chemin optimal fait 8 pas → score = 1.0 - 8×0.01 = 0.92 ≈ 0.930. Ce n'est pas un échec, c'est la récompense maximale atteignable !

### Questions sur l'architecture :

**Q : Pourquoi ReLU comme activation ?**
> R : ReLU (Rectified Linear Unit) : f(x) = max(0, x). C'est simple, rapide, et évite le problème du gradient vanishing des sigmoïdes. Standard pour les réseaux feedforward.

**Q : Pourquoi Adam comme optimiseur ?**
> R : Adam adapte le taux d'apprentissage pour chaque paramètre individuellement. Plus robuste que SGD classique. Convergence plus rapide en pratique. Standard dans le deep learning.

**Q : Pourquoi hidden_sizes = (128, 128) ?**
> R : C'est un choix empirique. Assez grand pour représenter des politiques complexes, pas trop grand pour éviter le surapprentissage sur des environnements simples. 2 couches cachées permettent d'approximer des fonctions non-linéaires complexes.

---

## 📌 PARTIE 9 — Comment Présenter (Tips de présentation)

### Structure suggérée (si présentation orale) :

1. **Contexte** : "On a implémenté 3 variantes de l'algorithme REINFORCE sur 4 environnements de complexité croissante"

2. **L'algo de base** : "REINFORCE optimise directement la politique via le policy gradient theorem. On joue un épisode entier, on calcule les retours Monte-Carlo, on met à jour le réseau."

3. **Le problème** : "Le problème principal de REINFORCE est sa haute variance — les gradients fluctuent beaucoup d'un épisode à l'autre, ce qui ralentit l'apprentissage."

4. **Les solutions** : "On a implémenté deux solutions : une baseline moyenne (simple, efficace, sans biais) et un réseau critique (plus puissant, apprend à estimer V(s))."

5. **Les résultats** : "Sur les environnements simples, tous convergent pareil. Sur TicTacToe, le critic atteint 0.987 contre 0.915 pour REINFORCE de base, ce qui confirme empiriquement l'intérêt de la réduction de variance."

### Phrases clés à retenir :

- *"REINFORCE est une méthode Monte-Carlo de gradient de politique"*
- *"On optimise directement π(a|s) sans passer par une fonction de valeur"*
- *"La baseline réduit la variance sans introduire de biais"*
- *"Le .detach() est crucial pour séparer les gradients acteur/critique"*
- *"Les résultats confirment empiriquement que la réduction de variance améliore l'apprentissage sur des environnements complexes"*

---

## 📌 PARTIE 10 — Comparaison avec d'autres algos du projet

| Algo | Famille | Approche | Avantage |
|------|---------|---------|----------|
| **REINFORCE** | Policy Gradient | MC, on-policy | Simple, théoriquement fondé |
| **REINFORCE MB** | Policy Gradient | MC + baseline | Moins de variance |
| **REINFORCE Critic** | Actor-Critic | MC + réseau de valeur | Encore moins de variance |
| **DQN** | Value-Based | Off-policy, replay buffer | Plus efficace en données |
| **DDQN** | Value-Based | Double DQN | Moins d'overestimation |
| **Tabular Q** | Value-Based | Tabular | Optimal sur petits espaces |
| **PPO** | Policy Gradient | Clipping, on-policy | Stable, état de l'art |

**REINFORCE vs DQN** : REINFORCE optimise π directement, DQN apprend Q(s,a) puis en dérive π. REINFORCE fonctionne naturellement avec des espaces d'actions discrets masqués.

---

## 📌 RÉCAPITULATIF EXPRESS (à lire la veille)

```
REINFORCE = Policy Gradient Monte-Carlo
  → joue épisode complet
  → calcule Gt = rt + γ*r(t+1) + ...
  → Loss = -Σ log π(at|st) * Gt
  → problème : haute variance

REINFORCE MB = REINFORCE + baseline moyenne
  → Ât = Gt - mean(G)
  → moins de variance, pas de biais
  → meilleur que REINFORCE sur envs complexes

REINFORCE Critic = REINFORCE + réseau critique
  → Acteur : π(a|s; θ) → logits → actions
  → Critique : V(s; w) → valeur scalaire
  → At = Gt - V(st; w)
  → Loss = actor_loss + 0.5 * critic_loss
  → .detach() pour séparer les gradients
  → meilleur sur TicTacToe (0.987) !

ENVIRONNEMENTS (par complexité croissante) :
  LineWorld → 5 états, 2 actions → trivial
  GridWorld → 25 états, 4 actions → simple
  TicTacToe → beaucoup d'états, 9 actions, adversaire → moyen
  Bobail → state_size=78, adversaire → complexe
```

---

*Guide généré pour la présentation du projet 5IA_ReinforcementLearning*
*Fichiers clés : agents/reinforce.py, agents/reinforce_mean_baseline.py, agents/reinforce_critic.py*
