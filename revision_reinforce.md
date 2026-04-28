# Fiche de révision complète — REINFORCE et ses variantes
## Ce qu'on a implémenté, pourquoi, et comment l'expliquer au prof

---

## 1. C'est quoi REINFORCE en une phrase ?

**REINFORCE c'est un agent qui apprend en jouant des parties complètes, et qui renforce les actions qui ont mené à de bons résultats.**

Contrairement à DQN (qui apprend à estimer la valeur de chaque action), REINFORCE apprend directement **quelle probabilité donner à chaque action** dans chaque situation.

---

## 2. Le problème que REINFORCE résout

Imagine un agent qui joue au TicTacToe. À chaque coup, il doit choisir une case. Comment sait-il si son coup était bon ?

- Avec DQN : on estime "si je joue ici, je gagnerai probablement X points"
- Avec REINFORCE : on joue la partie **jusqu'à la fin**, on regarde si on a gagné ou perdu, et on **remonte en arrière** pour dire "les coups qui ont mené à la victoire, je vais les faire plus souvent"

---

## 3. La politique π (pi) — ce que l'agent apprend

La **politique** $\pi_\theta(a|s)$ c'est une fonction qui dit :
> "Dans la situation $s$, quelle est la probabilité que je joue l'action $a$ ?"

C'est un **réseau de neurones** (PolicyNetwork) :
- **Entrée** : l'état du jeu (encodé en vecteur)
- **Couches cachées** : 2 couches de 128 neurones avec activation ReLU
- **Sortie** : des logits (un nombre par action possible)
- **Softmax** : transforme les logits en probabilités

**Le masquage des actions illégales** : avant le softmax, on met $-\infty$ sur les actions interdites. Ainsi leur probabilité devient 0 automatiquement. C'est important pour Bobail et TicTacToe où certains coups sont illégaux.

---

## 4. Le retour cumulé $G_t$ — comment on mesure si c'était bon

$$G_t = r_t + \gamma \cdot r_{t+1} + \gamma^2 \cdot r_{t+2} + \dots = \sum_{k=t}^{T} \gamma^{k-t} r_k$$

- $r_t$ = récompense reçue au pas $t$ (+1 victoire, -1 défaite, 0 sinon)
- $\gamma = 0.99$ = **facteur d'actualisation** : les récompenses futures valent légèrement moins que les immédiates
- $G_t$ est calculé **après la fin de l'épisode**, en remontant de la fin vers le début

**Pourquoi $\gamma < 1$ ?** Pour que l'agent préfère gagner vite plutôt que gagner tard. Aussi pour que la somme converge mathématiquement.

---

## 5. La mise à jour — comment l'agent apprend

La formule de mise à jour :
$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$$

En pratique, on **minimise la perte** (loss) :
$$\mathcal{L} = -\sum_{t=0}^{T} \log \pi_\theta(a_t|s_t) \cdot G_t$$

**Pourquoi le signe moins ?** PyTorch fait de la descente de gradient (minimise). Le théorème du gradient de politique dit qu'on veut monter le gradient → on minimise le négatif.

**Intuition** :
- Si $G_t > 0$ (bon épisode) → $-\log\pi \cdot G_t$ est négatif → minimiser ça **augmente** $\log\pi$ → **augmente** la probabilité de cette action
- Si $G_t < 0$ (mauvais épisode) → on **diminue** la probabilité

---

## 6. Pourquoi "Monte-Carlo" ?

Parce qu'on joue l'**épisode complet** avant de mettre à jour. On n'apprend pas à chaque pas (comme TD-learning ou DQN), on attend la fin pour avoir le vrai $G_t$.

**Avantage** : $G_t$ est une estimation non biaisée de la vraie valeur  
**Inconvénient** : **forte variance** — deux épisodes identiques peuvent donner des $G_t$ très différents selon ce qui se passe à la fin

---

## 7. Le problème de variance — pourquoi c'est grave

Imagine TicTacToe. L'agent joue le même coup au tour 1. Dans un épisode il gagne ($G_0 = 0.99$), dans un autre il perd ($G_0 = -1$). Le gradient change complètement selon l'épisode.

→ L'apprentissage est **instable et lent** car les mises à jour sont trop bruyantes.

**Solution** : soustraire une baseline pour réduire la variance.

---

## 8. REINFORCE Mean Baseline — la solution simple

Au lieu d'utiliser $G_t$ directement, on soustrait la **moyenne des retours de l'épisode** :

$$b = \frac{1}{T+1} \sum_{t=0}^{T} G_t$$

$$\mathcal{L} = -\sum_{t=0}^{T} \log \pi_\theta(a_t|s_t) \cdot (G_t - b)$$

**Intuition** : si tous les coups d'un épisode ont des retours au-dessus de la moyenne, on les renforce tous un peu. Si certains sont en dessous, on les affaiblit. On compare les coups **entre eux** plutôt qu'en absolu.

**Pourquoi ça ne biaise pas ?** Parce que $\mathbb{E}[\nabla \log\pi \cdot b] = 0$ pour n'importe quelle baseline constante. La moyenne de l'espérance ne change pas, seule la variance diminue.

**Résultats** : sur TicTacToe, ça passe de 0.915 à 0.939 à 100K épisodes. Sur Bobail, score stable à 1.000 dès 1K épisodes.

---

## 9. REINFORCE with Critic — la solution apprise

Au lieu d'une baseline constante (la moyenne), on apprend une baseline **adaptative** : le réseau critique $V(s; w)$.

**ValueNetwork** : même architecture que PolicyNetwork, mais la sortie est un **scalaire** — l'estimation de combien de récompense l'agent va accumuler depuis l'état $s$.

L'avantage :
$$A_t = G_t - V(s_t; w)$$

$A_t$ s'appelle l'**avantage** : "est-ce que ce coup était meilleur ou moins bon que ce que je m'attendais ?"

La perte totale :
$$\mathcal{L} = \underbrace{-\sum_t \log \pi_\theta(a_t|s_t) \cdot A_t}_{\text{acteur}} + 0.5 \cdot \underbrace{\sum_t (G_t - V(s_t;w))^2}_{\text{critique (MSE)}}$$

**Le coefficient 0.5** (`value_coef`) équilibre les deux pertes. C'est un hyperparamètre.

**Le `.detach()`** : quand on calcule la perte de l'acteur, on appelle `values_t.detach()`. Ça **coupe le gradient** : la perte de l'acteur ne met pas à jour le critique. Les deux réseaux apprennent de façon indépendante même avec un seul optimiseur.

**Un seul optimiseur Adam** pour les deux réseaux : `optim.Adam(params_acteur + params_critique, lr=0.001)`

**Résultats** : sur TicTacToe, 0.987 à 100K — le meilleur score. Le critique apprend $V(s)$ de mieux en mieux → avantages plus précis → moins de variance → meilleure politique.

---

## 10. Le gradient clipping — pourquoi on l'a mis

```python
torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
```

Parfois les gradients explosent (valeurs énormes) et détruisent les poids du réseau. Le clipping **plafonne** la norme du gradient à 1.0. C'est une mesure de sécurité standard dans les réseaux de neurones récurrents et de politique.

---

## 11. L'architecture complète de l'entraînement

```
Pour chaque épisode :
  1. reset() → état initial s_0
  2. Boucle jusqu'à fin de partie :
     - select_action(s, actions_légales) → échantillonne a ~ π_θ(·|s)
     - env.step(a) → s', r, done
     - stocker (log_prob, valeur_critique, récompense)
  3. Calculer G_t pour chaque pas (de la fin vers le début)
  4. Calculer la loss
  5. optimizer.zero_grad()
  6. loss.backward()
  7. clip_grad_norm_()
  8. optimizer.step()

Tous les 500 épisodes : évaluation greedy (pas de bruit, argmax)
```

---

## 12. BobailVsRandom — l'environnement d'entraînement

Pour Bobail, l'agent ne joue pas contre lui-même. Il joue contre un **adversaire aléatoire** (BobailVsRandom). 

Fonctionnement :
- L'agent est toujours **Joueur 0**
- Quand c'est le tour de l'adversaire (Joueur 1), le wrapper joue automatiquement un coup aléatoire
- L'agent ne voit que ses propres tours

**Pourquoi ?** Entraîner contre soi-même (self-play) est plus complexe. Un adversaire aléatoire permet d'apprendre les bases du jeu rapidement. Le score 1.000 signifie que l'agent bat l'adversaire aléatoire dans 100% des parties.

---

## 13. La correction du bobail.py — ce qui a changé

La prof a signalé un bug : **au premier tour, le Joueur 0 ne devait pas déplacer le Bobail**.

**Avant (bug)** : J0 commençait en phase 0 (déplacer Bobail) dès le premier tour  
**Après (corrigé)** : J0 commence en phase 1 (déplacer un pion) au premier tour seulement

Ce qui a changé dans le code :
- `state_size` : 77 → **78** (ajout d'un bit `first_turn` dans l'encodage de l'état)
- Le bit à l'indice 77 vaut 1.0 pendant le premier tour, 0.0 ensuite
- `reset()` démarre avec `phase=1` au lieu de `phase=0`

**Conséquence** : les modèles entraînés sur l'ancien bobail.py (state_size=77) sont **incompatibles** avec le nouveau. On a donc ré-entraîné REINFORCE Critic (le meilleur) sur le nouvel environnement. REINFORCE et REINFORCE MB pour Bobail ont aussi été ré-entraînés.

---

## 14. Les résultats — ce qu'il faut retenir

| Environnement | REINFORCE | MB | Critic |
|---|---|---|---|
| LineWorld | 1.000 | 1.000 | 1.000 |
| GridWorld | 0.930 | 0.930 | 0.930 |
| TicTacToe | 0.915 | 0.939 | **0.987** |
| Bobail | ~0.996 | 1.000 | ~0.996 |

**Ce qu'il faut retenir** :
- LineWorld/GridWorld : trop simples, tout le monde converge → pas discriminant
- TicTacToe : **l'environnement clé** qui montre la hiérarchie Critic > MB > REINFORCE
- Bobail : MB le plus stable, Critic le plus rapide (longueur d'épisode plus courte)

**Pourquoi GridWorld ne monte pas à 1.000 ?** À cause du facteur d'actualisation. Le chemin optimal est 8 pas. La récompense +1 à la fin, actualisée : $\gamma^8 = 0.99^8 \approx 0.923$. C'est cohérent avec 0.930.

---

## 15. Questions que le prof peut poser — et les réponses

**Q : Pourquoi REINFORCE est Monte-Carlo et pas TD ?**  
R : Parce qu'on attend la fin de l'épisode pour calculer $G_t$. TD (Temporal Difference) ferait une mise à jour à chaque pas avec une estimation bootstrap. Monte-Carlo utilise le vrai retour complet.

**Q : La baseline biaise-t-elle le gradient ?**  
R : Non. $\mathbb{E}_\pi[\nabla\log\pi(a|s) \cdot b] = b \cdot \mathbb{E}_\pi[\nabla\log\pi(a|s)] = b \cdot 0 = 0$. La baseline n'affecte pas l'espérance du gradient, seulement sa variance.

**Q : Pourquoi .detach() sur les valeurs du critique ?**  
R : Pour que la loss de l'acteur ($-\log\pi \cdot A_t$) ne propage pas de gradient dans le réseau critique. Sans detach, les paramètres $w$ du critique seraient mis à jour par deux sources simultanément, ce qui déstabilise l'apprentissage.

**Q : Pourquoi un seul optimiseur pour acteur et critique ?**  
R : C'est un choix d'implémentation. On concatène les paramètres des deux réseaux : `list(actor.parameters()) + list(critic.parameters())`. Adam gère les deux ensemble avec le même taux d'apprentissage.

**Q : Quelle est la différence entre REINFORCE et Actor-Critic (A2C/A3C) ?**  
R : Notre implémentation est un REINFORCE with baseline. Un vrai Actor-Critic (A2C) ferait des mises à jour à chaque pas (TD), pas seulement en fin d'épisode. Ici le critique sert uniquement de baseline, pas pour bootstrap les retours.

**Q : Pourquoi value_coef = 0.5 ?**  
R : C'est un hyperparamètre classique. Il équilibre l'importance relative des deux pertes. Si value_coef est trop grand, le critique domine et l'acteur n'apprend plus. Si trop petit, le critique n'apprend pas assez et la baseline est mauvaise.

**Q : Pourquoi le score Bobail Critic n'est pas stable à 1.000 ?**  
R : Parce qu'il a été ré-entraîné sur l'environnement corrigé (nouvelles règles, state_size=78). Le critique doit apprendre simultanément la nouvelle dynamique du jeu ET estimer $V(s)$ correctement. Avec plus d'épisodes, il convergerait probablement vers 1.000 stable. Il compense en ayant la stratégie la plus rapide (longueur 4.59).

**Q : Pourquoi ne pas utiliser l'entropie dans la loss ?**  
R : On aurait pu ajouter un terme d'entropie $-\beta \mathcal{H}(\pi)$ pour encourager l'exploration (comme dans PPO). On a fait le choix de garder l'implémentation proche du REINFORCE original de Williams (1992).

---

## 16. Les fichiers que tu as créés

| Fichier | Rôle |
|---|---|
| `agents/reinforce.py` | Agent REINFORCE pur (Monte-Carlo sans baseline) |
| `agents/reinforce_mean_baseline.py` | Agent REINFORCE avec baseline = moyenne épisode |
| `agents/reinforce_critic.py` | Agent REINFORCE avec réseau critique V(s) |
| `configs/config_reinforce.yaml` | Hyperparamètres REINFORCE (gamma, lr, hidden) |
| `configs/config_reinforce_mb.yaml` | Hyperparamètres REINFORCE MB (identiques) |
| `configs/config_reinforce_critic.yaml` | Hyperparamètres Critic (+ value_coef=0.5) |
| `train_reinforce.ps1` | Script PowerShell pour lancer les 12 entraînements |
| `envs/bobail.py` | Environnement Bobail corrigé (state_size=78) |
| `section_reinforce.tex` | Rapport LaTeX complet avec figures |
| `figures_reinforce/` | 4 graphiques générés par plot_reinforce.py |

---

## 17. La hiérarchie à retenir pour l'oral

> **Plus la baseline est précise → moins de variance → meilleure convergence**

1. **REINFORCE** : pas de baseline → haute variance → convergence lente
2. **REINFORCE MB** : baseline constante (moyenne) → variance réduite → convergence correcte  
3. **REINFORCE Critic** : baseline apprise $V(s)$ → variance minimale → meilleure convergence

Cette hiérarchie est **validée expérimentalement** par tes résultats sur TicTacToe (0.915 → 0.939 → 0.987).
