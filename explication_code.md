# Explication détaillée du code — Pour quelqu'un qui ne comprend rien
## Chaque ligne expliquée simplement

---

# FICHIER 1 : `agents/reinforce.py`

---

## Les imports (lignes 1-12)

```python
import numpy as np
```
NumPy = bibliothèque pour faire des maths sur des tableaux. On l'utilise pour calculer des moyennes (`np.mean`).

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
```
PyTorch = la bibliothèque qui gère les réseaux de neurones.
- `nn` = les couches du réseau (Linear, ReLU...)
- `optim` = les optimiseurs (Adam)
- `F` = des fonctions utiles (softmax, log_softmax, mse_loss)
- `Categorical` = une distribution de probabilité — permet de tirer un action au hasard selon des probabilités

```python
from torch.utils.tensorboard import SummaryWriter
```
TensorBoard = un outil qui crée des graphiques pendant l'entraînement. `SummaryWriter` écrit les données dans des fichiers que TensorBoard peut lire.

---

## La classe PolicyNetwork (le cerveau de l'agent)

```python
class PolicyNetwork(nn.Module):
```
On crée une classe qui **hérite** de `nn.Module`. C'est obligatoire pour tout réseau de neurones PyTorch. Ça donne accès à plein de fonctionnalités automatiquement (sauvegarde, transfert GPU, etc.)

```python
def __init__(self, state_size: int, action_size: int, hidden_sizes=(128, 128)):
    super().__init__()
```
`__init__` = le constructeur, appelé quand on fait `PolicyNetwork(78, 208)`.  
`super().__init__()` = appelle le constructeur de `nn.Module` pour initialiser correctement.

```python
    layers = []
    in_size = state_size
    for h in hidden_sizes:
        layers += [nn.Linear(in_size, h), nn.ReLU()]
        in_size = h
    layers.append(nn.Linear(in_size, action_size))
    self.net = nn.Sequential(*layers)
```
On **construit le réseau couche par couche** dynamiquement.

Pour Bobail avec hidden_sizes=(128,128), state_size=78, action_size=208 :
- `nn.Linear(78, 128)` → couche 1 : 78 entrées → 128 sorties
- `nn.ReLU()` → activation : met à 0 les valeurs négatives
- `nn.Linear(128, 128)` → couche 2 : 128 entrées → 128 sorties
- `nn.ReLU()` → activation
- `nn.Linear(128, 208)` → couche finale : 128 → 208 (un logit par action)

`nn.Sequential(*layers)` = empile toutes les couches dans un seul objet. L'étoile `*` "déplie" la liste.

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)
```
`forward` = ce qui se passe quand on "fait passer" un état dans le réseau.  
`x` entre → passe par toutes les couches → retourne les logits (208 nombres bruts).  
PyTorch appelle `forward` automatiquement quand on écrit `self._policy(state_t)`.

---

## La classe REINFORCE

```python
CHECKPOINTS = [1_000, 10_000, 100_000, 1_000_000]
```
Liste des épisodes où on évalue et sauvegarde. `1_000` = 1000 (le `_` est juste pour la lisibilité).

```python
def __init__(self, lr=1e-3, gamma=0.99, hidden_sizes=(128,128)):
    self.lr    = lr        # taux d'apprentissage
    self.gamma = gamma     # facteur d'actualisation
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self._policy    = None  # pas encore créé
    self._optimizer = None  # pas encore créé
```
`1e-3` = 0.001 en notation scientifique.  
`self.device` = détecte si un GPU est disponible. Si oui, utilise le GPU (plus rapide). Sinon, CPU.  
Le réseau est à `None` car on ne connaît pas encore la taille de l'état (ça dépend de l'environnement).

---

## `_ensure_network` — construction paresseuse

```python
def _ensure_network(self, state_size: int, action_size: int) -> None:
    if self._policy is not None:
        return
    self._policy    = PolicyNetwork(state_size, action_size, self.hidden_sizes).to(self.device)
    self._optimizer = optim.Adam(self._policy.parameters(), lr=self.lr)
```
**"Paresseuse"** = on crée le réseau seulement quand on en a besoin (au premier appel).  
`if self._policy is not None: return` → si déjà créé, on ne refait rien.  
`.to(self.device)` → envoie le réseau sur GPU si disponible.  
`optim.Adam(self._policy.parameters(), lr=0.001)` → crée l'optimiseur Adam qui va modifier les poids du réseau.

---

## `from_config` — créer l'agent depuis un fichier YAML

```python
@classmethod
def from_config(cls, config: dict) -> "REINFORCE":
    return cls(
        lr    = float(config.get("lr",    1e-3)),
        gamma = float(config.get("gamma", 0.99)),
        ...
    )
```
`@classmethod` = méthode qui s'appelle sur la **classe** et pas sur une instance.  
`config.get("lr", 1e-3)` = cherche `"lr"` dans le dict, retourne `1e-3` si absent.  
Concrètement : quand on lance `train.py`, le YAML est chargé en dict et passé ici.

---

## `select_action` — comment l'agent choisit une action

```python
def select_action(self, env, greedy=False):
    actions = env.available_actions()  # ex: [8, 16, 24, ...] pour Bobail

    state_t = torch.FloatTensor(env.get_state()).unsqueeze(0).to(self.device)
```
`env.get_state()` → retourne un numpy array de taille 78 (pour Bobail).  
`torch.FloatTensor(...)` → convertit en tenseur PyTorch.  
`.unsqueeze(0)` → ajoute une dimension : [78] devient [1, 78]. Le réseau attend un batch, même si c'est un seul état.

```python
    with torch.no_grad():
        logits = self._policy(state_t).squeeze(0)
```
`torch.no_grad()` → dit à PyTorch de ne PAS calculer les gradients ici. On ne s'entraîne pas, on choisit juste une action → économise de la mémoire.  
`.squeeze(0)` → enlève la dimension batch : [1, 208] redevient [208].  
`logits` = 208 nombres bruts (pas encore des probabilités).

```python
    mask = torch.full((env.action_size,), float("-inf"), device=self.device)
    for a in actions:
        mask[a] = 0.0
    probs = F.softmax(logits + mask, dim=-1)
```
**Le masquage** : on crée un vecteur de -∞ partout.  
Pour chaque action légale, on met 0 (au lieu de -∞).  
`logits + mask` → les actions illégales ont -∞ + n'importe quoi = -∞.  
`softmax(-∞)` = 0 → probabilité nulle pour les actions illégales. ✓

```python
    if greedy:
        return int(probs.argmax().item())
    return int(Categorical(probs).sample().item())
```
**Mode greedy** (évaluation/démo) : prend l'action avec la plus haute probabilité.  
**Mode exploration** (entraînement) : tire aléatoirement selon les probabilités.  
`Categorical(probs).sample()` → si probs=[0.7, 0.2, 0.1], tire l'action 0 dans 70% des cas.  
`.item()` → convertit le tenseur PyTorch en int Python.

---

## `_collect_episode` — jouer une partie entière

```python
def _collect_episode(self, env):
    state     = env.reset()   # réinitialise le jeu, retourne l'état initial
    log_probs = []            # stockera log(π(a_t|s_t)) pour chaque pas
    rewards   = []            # stockera r_t pour chaque pas

    while not env.is_game_over():
        actions = env.available_actions()
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits  = self._policy(state_t).squeeze(0)
```
Ici on n'a PAS `torch.no_grad()` → on garde le graphe de calcul pour pouvoir faire `loss.backward()` plus tard.

```python
        log_probs_all = F.log_softmax(logits + mask, dim=-1)
```
`log_softmax` = log(softmax(x)). Plus stable numériquement que `log(softmax(x))`.  
Résultat : un vecteur de 208 valeurs, chacune = log(probabilité de cette action).

```python
        action_t = Categorical(log_probs_all.exp()).sample()
        log_probs.append(log_probs_all[action_t])
```
`.exp()` = inverse du log → retrouve les probabilités.  
On tire une action, et on stocke le **log de sa probabilité** (pas la probabilité elle-même).  
Pourquoi le log ? Parce que la formule du gradient utilise `∇ log π`, pas `∇ π`.

```python
        state, reward, _ = env.step(action_t.item())
        rewards.append(reward)
```
On joue l'action, on récupère le nouvel état, la récompense, et si c'est fini.  
Le `_` ignore le booléen `done` (on gère ça avec `is_game_over()`).

---

## `_compute_returns` — calculer G_t en remontant vers l'arrière

```python
def _compute_returns(self, rewards):
    G       = 0.0
    returns = []
    for r in reversed(rewards):      # on part de la FIN de l'épisode
        G = r + self.gamma * G       # G_t = r_t + γ * G_{t+1}
        returns.insert(0, G)         # on insère au début (pour garder l'ordre)
    return returns
```

**Exemple concret** avec rewards = [0, 0, 1] et γ=0.99 :
- t=2 (fin) : G = 1 + 0.99 * 0 = **1.0**
- t=1 : G = 0 + 0.99 * 1.0 = **0.99**
- t=0 (début) : G = 0 + 0.99 * 0.99 = **0.9801**

Résultat : returns = [0.9801, 0.99, 1.0]

Chaque $G_t$ "sait" tout ce qui s'est passé après lui dans l'épisode.

---

## `_update_policy` — la mise à jour (le cœur de l'algo)

```python
def _update_policy(self, log_probs, returns):
    returns_t = torch.FloatTensor(returns).to(self.device)

    loss = torch.stack([-lp * G for lp, G in zip(log_probs, returns_t)]).sum()
```
`zip(log_probs, returns_t)` = parcourt les deux listes en parallèle.  
`-lp * G` = `-log(π(a_t|s_t)) * G_t` pour chaque pas.  
`.sum()` = somme sur tous les pas de l'épisode.  
Le signe **moins** car PyTorch minimise et on veut maximiser le gradient de politique.

```python
    self._optimizer.zero_grad()   # remet les gradients à 0 (sinon ils s'accumulent)
    loss.backward()               # calcule ∂loss/∂θ pour chaque paramètre du réseau
    torch.nn.utils.clip_grad_norm_(self._policy.parameters(), max_norm=1.0)
    self._optimizer.step()        # met à jour les poids : θ ← θ - lr * ∂loss/∂θ
```
`zero_grad()` → OBLIGATOIRE avant chaque mise à jour. Sans ça, les gradients du dernier épisode s'ajoutent à ceux du suivant.  
`backward()` → la magie de PyTorch : remonte automatiquement le graphe de calcul et calcule tous les gradients.  
`clip_grad_norm_` → si la norme du gradient > 1.0, la ramène à 1.0. Évite les explosions.  
`step()` → applique réellement la mise à jour des poids.

---

## `get_state_dict` / `load_state_dict` — sauvegarder et charger

```python
def get_state_dict(self):
    return {
        "policy": self._policy.state_dict(),  # les poids du réseau
        "lr":     self.lr,
        "gamma":  self.gamma,
        ...
    }
```
`state_dict()` = dictionnaire PyTorch contenant tous les poids et biais du réseau. C'est ça qui est sauvegardé dans les fichiers `.pt`.

```python
def load_state_dict(self, data, state_size, action_size):
    self._ensure_network(state_size, action_size)  # recrée le réseau avec la bonne taille
    self._policy.load_state_dict(data["policy"])   # charge les poids sauvegardés
```
Quand la GUI charge un modèle, elle appelle ça pour restaurer exactement l'état de l'agent au moment de la sauvegarde.

---
---

# FICHIER 2 : `agents/reinforce_mean_baseline.py`

Ce fichier est **identique** à `reinforce.py` sauf une seule méthode : `_update_policy`.

## La seule différence : `_update_policy`

```python
# REINFORCE pur :
loss = torch.stack([-lp * G for lp, G in zip(log_probs, returns_t)]).sum()

# REINFORCE Mean Baseline :
baseline   = returns_t.mean()        # moyenne des G_t de l'épisode
advantages = returns_t - baseline    # A_t = G_t - moyenne
loss = torch.stack([-lp * adv for lp, adv in zip(log_probs, advantages)]).sum()
```

**Concrètement** avec returns = [0.98, 0.99, 1.0] :
- baseline = (0.98 + 0.99 + 1.0) / 3 = **0.99**
- advantages = [0.98-0.99, 0.99-0.99, 1.0-0.99] = **[-0.01, 0.00, +0.01]**

Le dernier coup (+0.01) est légèrement renforcé, le premier (-0.01) légèrement affaibli. Les mises à jour sont beaucoup plus petites et stables.

---
---

# FICHIER 3 : `agents/reinforce_critic.py`

---

## Les deux réseaux

```python
class PolicyNetwork(nn.Module):   # identique à reinforce.py → état → logits
class ValueNetwork(nn.Module):    # NOUVEAU → état → scalaire V(s)
```

```python
class ValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_sizes=(128,128)):
        ...
        layers.append(nn.Linear(in_size, 1))   # sortie = 1 seul neurone
    
    def forward(self, x):
        return self.net(x).squeeze(-1)   # [batch, 1] → [batch]
```
La sortie `1` = un seul nombre = l'estimation de combien l'agent va gagner depuis cet état.  
`.squeeze(-1)` enlève la dernière dimension inutile.

---

## `_ensure_networks` — deux réseaux, un seul optimiseur

```python
def _ensure_networks(self, state_size, action_size):
    self._actor  = PolicyNetwork(state_size, action_size, ...).to(self.device)
    self._critic = ValueNetwork(state_size, ...).to(self.device)
    
    params = list(self._actor.parameters()) + list(self._critic.parameters())
    self._optimizer = optim.Adam(params, lr=self.lr)
```
On **concatène** les paramètres des deux réseaux dans une seule liste.  
Adam va mettre à jour les deux réseaux en même temps avec un seul appel à `.step()`.

---

## `_collect_episode` — différence clé avec REINFORCE pur

```python
# Dans REINFORCE pur :
log_probs, rewards = self._collect_episode(env)

# Dans REINFORCE Critic :
log_probs, values, rewards = self._collect_episode(env)
#          ^^^^^^ NOUVEAU : on stocke aussi V(s_t) pour chaque pas
```

```python
while not env.is_game_over():
    # Acteur (identique à REINFORCE pur)
    logits = self._actor(state_t).squeeze(0)
    ...
    log_probs.append(log_probs_all[action_t])

    # Critique (NOUVEAU)
    value = self._critic(state_t).squeeze(0)
    values.append(value)          # on garde le tenseur AVEC le graphe de calcul
```

**IMPORTANT** : on n'écrit pas `value.detach()` ici. On garde le lien avec le graphe pour que `loss.backward()` puisse mettre à jour le critique.

---

## `_update` — le cœur du Critic, la partie la plus importante

```python
def _update(self, log_probs, values, returns):
    returns_t = torch.FloatTensor(returns).to(self.device)
    values_t  = torch.stack(values)   # empile les tenseurs en une matrice
```

```python
    # PERTE DU CRITIQUE : le critique doit prédire G_t le plus précisément possible
    critic_loss = F.mse_loss(values_t, returns_t)
    # = moyenne de (V(s_t) - G_t)²  pour chaque pas
```
Si le critique prédit V(s) = 0.8 et que G_t = 1.0, la perte est (0.8-1.0)² = 0.04. Il va se corriger.

```python
    # AVANTAGE : G_t - V(s_t) mais on DÉTACHE les valeurs du critique
    advantages = returns_t - values_t.detach()
    #                                 ^^^^^^^^ CRUCIAL
```
`.detach()` = coupe le lien entre `advantages` et le graphe de calcul du critique.  

**Sans `.detach()`** : quand on fait `actor_loss.backward()`, PyTorch remonterait le gradient jusqu'au critique et modifierait ses poids pour minimiser la perte de l'acteur. Le critique apprendrait à **aider l'acteur à minimiser sa loss** plutôt qu'à **estimer V(s)** correctement. Les deux réseaux se parasiteraient.  

**Avec `.detach()`** : l'acteur utilise les valeurs du critique comme des constantes. Il ne peut pas les modifier. Chaque réseau a son propre objectif indépendant.

```python
    actor_loss = torch.stack([-lp * adv for lp, adv in zip(log_probs, advantages)]).sum()
```
Identique à REINFORCE MB mais avec `adv = G_t - V(s_t)` au lieu de `G_t - mean(G)`.

```python
    total_loss = actor_loss + self.value_coef * critic_loss
    #                         ^^^^^^^^^^^^^^^ = 0.5
```
On additionne les deux pertes. Le facteur 0.5 dit "le critique est deux fois moins important que l'acteur". Si value_coef = 2, le critique dominerait et l'acteur n'apprendrait plus rien.

```python
    self._optimizer.zero_grad()
    total_loss.backward()          # calcule les gradients pour TOUS les paramètres
    clip_grad_norm_(all_params, max_norm=1.0)
    self._optimizer.step()         # met à jour acteur ET critique en même temps
```

---

## Différence get_state_dict entre les 3 agents

```python
# REINFORCE pur sauvegarde :
{"policy": ..., "lr": ..., "gamma": ...}

# REINFORCE MB sauvegarde pareil :
{"policy": ..., "lr": ..., "gamma": ...}

# REINFORCE Critic sauvegarde :
{"actor": ..., "critic": ..., "lr": ..., "gamma": ..., "value_coef": ...}
```

C'est pour ça que la GUI plante si on donne le mauvais fichier : elle cherche la clé `"policy"` pour MB mais le fichier Critic a `"actor"` et `"critic"`.

---
---

# FICHIER 4 : `envs/bobail.py` — les changements importants

## Avant la correction (bug)

```python
def reset(self):
    self._phase = 0    # J0 commence en phase 0 = déplacer le Bobail
```
J0 devait déplacer le Bobail dès le premier tour. Ce n'était pas les vraies règles.

## Après la correction (nouveau fichier de la prof)

```python
def __init__(self):
    self._phase      = 1     # J0 commence en phase 1 = déplacer un pion
    self._first_turn = True  # marqueur du premier tour

def reset(self):
    self._phase      = 1     # premier tour toujours en phase 1
    self._first_turn = True
```

```python
# Dans step(), quand J0 joue son premier pion :
if self._first_turn:
    self._first_turn = False   # désactive le marqueur
self._end_turn()               # passe à J1, phase 0 (J1 déplace le Bobail normalement)
```

## L'état encodé : state_size 77 → 78

```python
def get_state(self):
    state = np.zeros(78, dtype=np.float32)   # était 77
    ...
    state[75] = float(self._current_player)
    state[76] = float(self._phase)
    state[77] = 1.0 if self._first_turn else 0.0   # NOUVEAU BIT
    return state
```

Le réseau de neurones doit savoir si c'est le premier tour pour adapter son comportement. Sans ce bit, l'état "phase 1, tour 1" serait identique à "phase 1, tour 50" — mais les actions légales sont différentes.

**Conséquence** : les poids sauvegardés avec state_size=77 ont une couche d'entrée `Linear(77, 128)`. Si on essaie de charger dans un réseau `Linear(78, 128)`, les dimensions ne correspondent pas → erreur PyTorch.

---
---

# RÉSUMÉ VISUEL — Ce qui se passe pendant un épisode complet

```
REINFORCE pur :

Episode 1 :
  reset() → s_0
  s_0 → PolicyNet → logits → softmax+mask → probs → tirer a_0 → stocker log(π(a_0|s_0))
  step(a_0) → s_1, r_0=0
  s_1 → PolicyNet → ... → tirer a_1 → stocker log(π(a_1|s_1))
  step(a_1) → s_2, r_1=0
  ...
  step(a_T) → s_fin, r_T=+1, done=True

  Calculer G_T = 1.0
  Calculer G_{T-1} = 0 + 0.99 * 1.0 = 0.99
  ...

  loss = -log(π(a_0|s_0))*G_0 - log(π(a_1|s_1))*G_1 - ... - log(π(a_T|s_T))*G_T
  loss.backward() → gradients
  optimizer.step() → nouveaux poids

Episode 2 : recommencer avec les nouveaux poids
```

```
REINFORCE Critic — différence dans la mise à jour :

  Pendant l'épisode, à chaque pas on stocke AUSSI V(s_t) = CriticNet(s_t)
  
  Après l'épisode :
  critic_loss = moyenne((V(s_0)-G_0)², (V(s_1)-G_1)², ...)
  advantages  = [G_0-V(s_0), G_1-V(s_1), ...]  (valeurs détachées)
  actor_loss  = -log(π(a_0|s_0))*adv_0 - log(π(a_1|s_1))*adv_1 - ...
  total_loss  = actor_loss + 0.5 * critic_loss
  total_loss.backward() → met à jour acteur ET critique
```

---

# POINTS À RETENIR ABSOLUMENT

1. **`log_probs`** stocke des tenseurs PyTorch **avec** graphe de calcul → permet `backward()`
2. **`values`** dans le Critic aussi stockés avec graphe → mais **détachés** pour la perte acteur
3. **`zero_grad()`** toujours avant `backward()` sinon les gradients s'accumulent
4. **`clip_grad_norm_`** plafonne les gradients → évite que les poids explosent
5. **Le `.pt`** contient les poids du réseau → incompatible si l'architecture change (ex: 77→78)
6. **Mode greedy** (`argmax`) pour évaluation et démo GUI → pas de hasard
7. **Mode stochastique** (`Categorical.sample()`) pour l'entraînement → exploration
