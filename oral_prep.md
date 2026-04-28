# Préparation orale complète — REINFORCE
## Tout ce qu'il faut savoir, expliqué depuis zéro

---

# PARTIE 1 — COMPRENDRE LE CONTEXTE : C'EST QUOI L'APPRENTISSAGE PAR RENFORCEMENT ?

## L'idée de base en une analogie

Imagine que tu apprends à jouer aux fléchettes. Personne ne t'explique comment tenir la fléchette, comment viser. Tu essaies, tu rates, tu essaies encore. Quand tu marques un point, ton cerveau dit "ah, ce geste était bien, je vais le refaire plus souvent". Quand tu rates, "ce geste était mauvais, je vais éviter de le refaire".

**C'est exactement ça l'apprentissage par renforcement (RL).**

- L'agent = toi qui joues aux fléchettes
- L'environnement = la cible et les règles
- L'action = lancer la fléchette d'une certaine façon
- La récompense = les points marqués (ou perdus)
- La politique = ta stratégie ("dans telle situation, je fais tel geste")

## Les deux grandes familles de méthodes RL

**Méthodes basées sur la valeur (DQN, DDQN...)** :
> "Je vais estimer combien vaut chaque action dans chaque situation, et choisir la meilleure."

C'est comme si tu mémorisais pour chaque position de ton bras, le score que tu vas probablement marquer.

**Méthodes de gradient de politique (REINFORCE...)** :
> "Je vais directement apprendre ma stratégie — quelle probabilité donner à chaque action."

C'est comme si tu apprenais directement le bon geste sans passer par une estimation.

**REINFORCE fait partie de la deuxième famille.**

---

# PARTIE 2 — REINFORCE EXPLIQUÉ SIMPLEMENT

## L'idée en 3 phrases

1. L'agent joue une partie complète du début à la fin
2. À la fin, il regarde ce qui s'est passé
3. Il augmente la probabilité des actions qui ont mené à gagner, diminue celles qui ont mené à perdre

## Pourquoi "Monte-Carlo" ?

Monte-Carlo = on attend la fin pour avoir le résultat réel.

Comme au casino : tu joues jusqu'à la fin de la nuit, puis tu calcules si tu as gagné ou perdu. Tu n'essaies pas de prédire à mi-chemin ce qui va se passer.

**Opposé** : TD-learning (Temporal Difference) = on prédit à chaque pas. Comme si après chaque mise au casino tu calculais "avec la chance que j'ai eu jusqu'ici, je vais probablement finir à +100€".

## La politique π (comment l'agent décide)

La politique `π(a|s)` = "dans la situation `s`, quelle est la probabilité que je joue l'action `a` ?"

Exemple TicTacToe, état = plateau avec 3 cases occupées :
```
π(jouer case 7 | état) = 0.45
π(jouer case 2 | état) = 0.30
π(jouer case 5 | état) = 0.25
```

Ces probabilités sont calculées par le réseau de neurones. Au début de l'entraînement elles sont aléatoires (~1/N pour N actions). À la fin elles reflètent la vraie stratégie apprise.

## Le retour G_t — mesurer si c'était bien

G_t = "combien vais-je accumuler comme récompense depuis ce moment jusqu'à la fin ?"

```
Épisode : [coup1, coup2, coup3, VICTOIRE]
Récompenses : [0,    0,    0,    +1]

G_2 = 0 + 0.99 * 1     = 0.99  (proche de la victoire)
G_1 = 0 + 0.99 * 0.99  = 0.98  (un peu plus loin)
G_0 = 0 + 0.99 * 0.98  = 0.97  (encore plus loin)
```

Le γ = 0.99 fait que les récompenses **proches dans le temps** valent plus. Si on avait γ = 1, tous les coups auraient le même poids. Si γ = 0, seule la récompense immédiate compte.

**Intuition γ = 0.99** : gagner en 2 coups est légèrement mieux que gagner en 10 coups. L'agent préfère finir vite.

## La mise à jour — comment ça apprend

La formule :
```
θ ← θ + α * Σ ∇log π(a_t|s_t) * G_t
```

Traduit en français :
```
nouveaux_poids = anciens_poids + taux_apprentissage * (direction qui maximise)
```

Si G_t > 0 (bon résultat) : on augmente `log π(a_t|s_t)` → on augmente la probabilité de `a_t`
Si G_t < 0 (mauvais résultat) : on diminue `log π(a_t|s_t)` → on diminue la probabilité de `a_t`

**Analogie** : après une bonne partie de TicTacToe, tu dis "le coup que j'ai joué en position 3 était super, je vais le rejouer plus souvent dans cette situation".

---

# PARTIE 3 — LE PROBLÈME DE VARIANCE ET LES SOLUTIONS

## C'est quoi le problème de variance ?

Imagine TicTacToe. Tu joues le même premier coup (case centrale) dans deux parties :
- Partie 1 : tu gagnes → G_0 = +0.97 → "super coup !"
- Partie 2 : tu perds → G_0 = -0.97 → "mauvais coup !"

Mais c'est le **même coup** dans la **même situation** ! Le résultat dépend de ce qui se passe après, pas seulement du premier coup.

→ Les mises à jour sont **incohérentes** et **bruyantes**
→ L'apprentissage est **lent** car l'agent reçoit des signaux contradictoires

**C'est le problème de haute variance.**

## Solution 1 : Baseline Moyenne (Mean Baseline)

Au lieu de juger chaque coup par rapport à 0 (+/- selon victoire/défaite), on juge chaque coup **par rapport aux autres coups du même épisode**.

```
returns = [0.97, 0.98, 0.99]  (une partie gagnée en 3 coups)
moyenne = 0.98

advantages = [0.97-0.98, 0.98-0.98, 0.99-0.98]
           = [-0.01,     0.00,      +0.01]
```

Maintenant les mises à jour sont **beaucoup plus petites** et **stables**.
Le premier coup est légèrement pénalisé (il était "moins bien" que la moyenne de l'épisode).
Le dernier coup est légèrement renforcé (il était "le meilleur" du lot).

**Analogie** : au lieu de dire "j'ai eu 15/20, c'est bien ou pas ?", on dit "j'ai eu 15/20 alors que la moyenne de la promo était 14/20, donc c'est au-dessus de la moyenne".

## Solution 2 : Réseau Critique (Actor-Critic)

Au lieu d'une baseline constante (la moyenne de l'épisode), on apprend une baseline **intelligente** : le réseau critique V(s).

V(s) = "depuis cet état s, combien vais-je probablement gagner ?"

Si V(s) = 0.9 et que G_t = 0.95 → avantage = +0.05 (légèrement mieux que prévu)
Si V(s) = 0.9 et que G_t = 0.5 → avantage = -0.4 (beaucoup moins bien que prévu)

**Analogie** : tu joues à la belote. Avant chaque coup, tu estimes "avec mes cartes, je vais probablement gagner ce tour". Si tu gagnes plus que prévu → excellent coup. Si tu gagnes moins → mauvais coup.

Plus le critique est précis dans ses estimations, plus les avantages sont précis, moins il y a de variance.

## Hiérarchie des 3 méthodes

```
REINFORCE pur          → variance HAUTE  → convergence LENTE
REINFORCE Mean Baseline → variance MOYENNE → convergence CORRECTE  
REINFORCE Critic       → variance FAIBLE → convergence RAPIDE
```

Tes résultats sur TicTacToe le confirment :
```
REINFORCE pur          : 0.915 à 100K épisodes
REINFORCE Mean Baseline : 0.939 à 100K épisodes  (+2.4 pts)
REINFORCE Critic       : 0.987 à 100K épisodes  (+7.2 pts)
```

---

# PARTIE 4 — LE RÉSEAU DE NEURONES EN DÉTAIL

## C'est quoi un réseau de neurones ?

Un réseau de neurones = une fonction mathématique très complexe qui transforme une entrée en sortie.

Pour REINFORCE :
```
ENTRÉE : état du jeu (vecteur de nombres)
   ↓
[Couche 1 : 128 neurones, ReLU]
   ↓
[Couche 2 : 128 neurones, ReLU]
   ↓
SORTIE : un nombre par action possible (logits)
```

## C'est quoi un "logit" ?

Un logit = un nombre brut avant transformation en probabilité.

Exemple : le réseau sort [2.1, -0.5, 1.3] pour 3 actions.
Ce ne sont pas des probabilités (elles ne somment pas à 1).

On applique **softmax** pour convertir en probabilités :
```
softmax([2.1, -0.5, 1.3]) → [0.67, 0.05, 0.28]
```
Maintenant ça somme à 1. ✓

## C'est quoi ReLU ?

ReLU = Rectified Linear Unit = une fonction très simple :
```
ReLU(x) = x si x > 0
ReLU(x) = 0 si x ≤ 0
```

Pourquoi ? Les réseaux sans activation sont juste des matrices (opérations linéaires). ReLU introduit de la **non-linéarité** = capacité d'apprendre des patterns complexes.

## C'est quoi Adam ?

Adam = un algorithme d'optimisation plus intelligent que la descente de gradient classique.

Descente de gradient basique : `θ ← θ - lr * gradient`
Adam : adapte le taux d'apprentissage individuellement pour chaque paramètre. Les paramètres qui bougent peu reçoivent un plus grand pas. Ceux qui bougent beaucoup reçoivent un plus petit pas. Plus stable et plus rapide en pratique.

## Le masquage des actions illégales

TicTacToe : on ne peut pas jouer sur une case déjà occupée.
Bobail : certains déplacements sont impossibles.

Sans masquage, le réseau pourrait choisir une action illégale et crasher l'environnement.

Solution :
```python
mask = [-∞, -∞, -∞, -∞, -∞, -∞, -∞, -∞, -∞]  # tout illégal par défaut
pour chaque action légale:
    mask[action] = 0

probs = softmax(logits + mask)
```

`softmax(-∞)` = 0 → probabilité nulle pour les actions illégales. Impossible de les choisir.

---

# PARTIE 5 — L'ENVIRONNEMENT BOBAIL

## Les règles (version simplifiée)

- Plateau 5×5
- Joueur 0 : 5 pions en bas (ligne 4)
- Joueur 1 : 5 pions en haut (ligne 0)
- 1 Bobail (pièce neutre) au centre

**Chaque tour (sauf le tout premier de J0) :**
1. Déplacer le Bobail d'une case (8 directions)
2. Déplacer un de ses pions (glisse jusqu'au mur/obstacle)

**Gagner :**
- Amener le Bobail sur sa propre ligne de base
- Bloquer le Bobail (adversaire ne peut plus le bouger)

**Perdre :**
- Ne pas pouvoir bouger le Bobail en début de tour

## L'encodage de l'état (state_size = 78)

L'état = un vecteur de 78 nombres :
```
Indices 0-24  : positions des pions de J0 (1 si J0 est là, 0 sinon)
Indices 25-49 : positions des pions de J1
Indices 50-74 : position du Bobail
Indice 75     : joueur courant (0 ou 1)
Indice 76     : phase courante (0=bouger Bobail, 1=bouger pion)
Indice 77     : premier tour ? (1 si oui, 0 sinon)
```

Exemple : si J0 a un pion en (0,0) → case 0 → state[0] = 1.0

## L'encodage des actions (action_size = 208)

Actions 0-7 : déplacer le Bobail dans les 8 directions (phase 0)
Actions 8-207 : déplacer un pion (phase 1)

Pour déplacer le pion en (2,3) vers le nord (direction 0) :
```
case_idx = 2*5 + 3 = 13
action   = 8 + 13*8 + 0 = 112
```

## BobailVsRandom — pourquoi ?

Notre agent joue toujours en J0.
L'adversaire (J1) joue des coups aléatoires automatiquement.

Pourquoi pas contre lui-même ? Le self-play est plus complexe (l'environnement d'entraînement change en même temps que l'agent apprend). L'adversaire aléatoire permet d'apprendre les bases efficacement.

Score 1.000 = l'agent bat J1 aléatoire dans 100% des parties.

---

# PARTIE 6 — LES RÉSULTATS ET CE QU'ILS SIGNIFIENT

## LineWorld — Score 1.000 pour tout le monde

**C'est quoi LineWorld ?** Une ligne de cases. L'agent est au centre, doit aller à droite en 2 pas.

**Pourquoi 1.000 dès 1000 épisodes ?** L'espace d'état est minuscule (7 cases, 2 actions). N'importe quelle méthode apprend en quelques centaines d'épisodes. Pas de différence entre les 3 variantes.

**Longueur = 2.0** = l'agent prend toujours le chemin optimal (2 pas).

## GridWorld — Score 0.930 pour tout le monde

**C'est quoi GridWorld ?** Une grille 2D avec des murs. L'agent doit trouver le chemin le plus court.

**Pourquoi 0.930 et pas 1.000 ?** À cause du facteur d'actualisation γ=0.99. Le chemin optimal est 8 pas. La récompense finale (+1) actualisée : 0.99^8 ≈ 0.923. C'est cohérent avec 0.930.

**Pourquoi pas de différence entre les 3 ?** Comme LineWorld, environnement trop simple. La variance ne joue pas de rôle.

## TicTacToe — Les 3 variantes se distinguent clairement

**C'est quoi TicTacToe ?** Morpion, 3×3. L'agent joue contre un adversaire aléatoire.

**Pourquoi c'est plus complexe ?** Plus d'états possibles, présence d'un adversaire (imprévisible), horizon plus long.

**REINFORCE pur 0.915** : converge lentement. La variance haute fait que l'agent met beaucoup d'épisodes à comprendre quels coups sont vraiment bons.

**REINFORCE MB 0.939** : la baseline réduit le bruit. L'agent comprend plus vite quels coups sont bons *relativement aux autres coups du même épisode*.

**REINFORCE Critic 0.987** : quasi-parfait. Le critique apprend à estimer "depuis cette position, je vais probablement gagner". Les avantages sont très précis → mises à jour très efficaces. La longueur moyenne 3.164 montre que l'agent gagne en 3 coups en moyenne (très rapide).

## Bobail — Résultats spécifiques

**REINFORCE 0.996** : jamais stable à 1.000. La variance fait qu'il perd occasionnellement même après 100K épisodes.

**REINFORCE MB 1.000** : parfait et stable. La baseline suffit sur Bobail car l'adversaire est aléatoire (environnement relativement prévisible).

**REINFORCE Critic 0.996** : ré-entraîné sur le nouvel environnement (state_size=78). Le critique apprend la nouvelle dynamique + V(s) simultanément. Légèrement moins stable que MB mais longueur plus courte (4.59 vs 5.23) → stratégie plus agressive.

---

# PARTIE 7 — QUESTIONS ORALES AVEC RÉPONSES DÉTAILLÉES

## Questions sur REINFORCE de base

**Q : Qu'est-ce que REINFORCE ?**

R : REINFORCE est un algorithme de gradient de politique proposé par Williams en 1992. Il optimise directement une politique stochastique π_θ(a|s) en utilisant des épisodes complets (approche Monte-Carlo). La mise à jour renforce les actions proportionnellement au retour cumulé G_t qu'elles ont généré. C'est l'algorithme fondateur des méthodes policy gradient modernes comme PPO et A3C.

---

**Q : Pourquoi utiliser un réseau de neurones pour représenter la politique ?**

R : Un tableau (comme en Q-learning tabulaire) n'est pas scalable. Pour Bobail, l'espace d'état est de dimension 78 → potentiellement des milliards d'états distincts. Un réseau de neurones généralise : il apprend des patterns généraux ("quand le Bobail est proche de ma ligne de base, jouer dans cette direction") plutôt que mémoriser chaque état.

---

**Q : C'est quoi le théorème du gradient de politique ?**

R : Il établit que le gradient de l'espérance du retour cumulé par rapport aux paramètres θ est :
∇_θ J(θ) = E[Σ_t ∇_θ log π_θ(a_t|s_t) · G_t]

Ce théorème est crucial car il dit qu'on peut estimer le gradient **sans connaître le modèle de l'environnement** (transitions, récompenses). On a juste besoin d'échantillonner des trajectoires.

---

**Q : Pourquoi γ < 1 ?**

R : Trois raisons :
1. **Mathématique** : assure la convergence de la somme infinie des récompenses
2. **Économique** : une récompense maintenant vaut plus qu'une récompense future (préférence temporelle)
3. **Pratique** : encourage l'agent à gagner vite plutôt que lentement. γ=0.99 sur 8 pas = 0.99^8 ≈ 0.923, donc l'agent préfère gagner en 2 pas (0.99^2 ≈ 0.980) plutôt qu'en 8 pas.

---

## Questions sur la variance et les baselines

**Q : C'est quoi la variance dans REINFORCE et pourquoi est-ce un problème ?**

R : La variance mesure à quel point les mises à jour du gradient varient d'un épisode à l'autre. Dans REINFORCE, G_t dépend de tout l'épisode — une seule récompense aléatoire en fin de partie peut changer radicalement G_t. Deux épisodes avec les mêmes premiers coups peuvent donner des G_t très différents. Résultat : les mises à jour sont incohérentes, l'apprentissage est lent et instable.

---

**Q : Pourquoi la baseline ne biaise pas le gradient ?**

R : Parce que E_π[∇ log π(a|s) · b] = 0 pour n'importe quelle baseline b indépendante de l'action. Preuve :
E[∇ log π · b] = Σ_a π(a|s) · ∇ log π(a|s) · b = b · Σ_a ∇ π(a|s) = b · ∇ Σ_a π(a|s) = b · ∇1 = 0

Donc soustraire n'importe quelle baseline ne change pas l'espérance du gradient. Ça réduit la variance sans introduire de biais.

---

**Q : Quelle est la différence entre REINFORCE MB et REINFORCE Critic ?**

R : 
- **REINFORCE MB** : baseline = moyenne des retours de l'épisode courant. C'est une constante, simple à calculer, pas de réseau supplémentaire.
- **REINFORCE Critic** : baseline = V(s_t) estimé par un réseau de neurones. La baseline est différente pour chaque état et s'améliore au fil du temps. Plus précise → moins de variance. Mais nécessite un second réseau + équilibrage des deux pertes.

---

**Q : Pourquoi .detach() dans la perte de l'acteur ?**

R : Quand on calcule `advantages = G_t - V(s_t)` et qu'on fait `actor_loss.backward()`, PyTorch remonte le graphe de calcul. Sans `.detach()`, le gradient de l'acteur traverserait V(s_t) et modifierait les poids du critique pour minimiser la perte de l'acteur — ce qui n'est pas l'objectif du critique. Avec `.detach()`, les valeurs du critique sont utilisées comme des constantes du point de vue de l'acteur. Les deux réseaux ont des objectifs indépendants et cohérents.

---

**Q : Pourquoi un seul optimiseur pour acteur et critique ?**

R : C'est un choix d'implémentation. On concatène les paramètres des deux réseaux : `params = list(actor.parameters()) + list(critic.parameters())`. Adam maintient des statistiques de gradient pour chaque paramètre séparément. Avoir un seul optimiseur est équivalent à deux optimiseurs séparés avec le même lr, mais plus simple à coder.

---

**Q : C'est quoi le gradient clipping et pourquoi ?**

R : `clip_grad_norm_(params, max_norm=1.0)` plafonne la norme L2 du vecteur gradient à 1.0. Si le gradient est trop grand (explosion de gradient), les poids seraient mis à jour de façon catastrophique. Ça arrive souvent dans les réseaux de politique car les trajectoires peuvent avoir des G_t très variables. Le clipping stabilise l'entraînement sans changer la direction du gradient, juste son amplitude.

---

## Questions sur les environnements

**Q : Pourquoi les résultats de LineWorld et GridWorld sont identiques pour les 3 variantes ?**

R : Ces environnements ont un espace d'état très restreint et aucun adversaire (dynamique déterministe). La variance n'est pas un facteur limitant car les épisodes sont courts et prévisibles. N'importe quelle méthode de gradient de politique converge en quelques centaines d'épisodes. La réduction de variance apportée par les baselines n'apporte aucun avantage mesurable sur des problèmes aussi simples.

---

**Q : Pourquoi REINFORCE Critic est le meilleur sur TicTacToe mais pas nécessairement sur Bobail ?**

R : Sur TicTacToe, l'environnement est stable (même règles tout au long de l'entraînement). Le critique a 100K épisodes pour apprendre V(s) correctement, ce qui lui donne un avantage clair.

Sur Bobail, REINFORCE Critic a été ré-entraîné sur un environnement corrigé (nouvelles règles, state_size=78) alors que REINFORCE MB a été entraîné sur l'ancienne version. La comparaison n'est pas parfaitement équitable. De plus, REINFORCE MB suffit pour battre un adversaire aléatoire — le problème n'est pas assez difficile pour que le critique fasse une différence en termes de score. Le Critic se distingue uniquement par la longueur d'épisode (stratégie plus rapide).

---

**Q : Qu'est-ce que BobailVsRandom et pourquoi ce choix ?**

R : BobailVsRandom est un wrapper autour de l'environnement Bobail qui joue automatiquement les coups de l'adversaire (Joueur 1) de façon aléatoire. L'agent (Joueur 0) ne voit jamais les tours de l'adversaire — il reçoit directement l'état après les coups adverses.

Ce choix simplifie l'entraînement : avec un adversaire fixe (aléatoire), l'environnement est stationnaire. Le self-play (jouer contre soi-même) crée un environnement non-stationnaire qui nécessite des techniques plus avancées (comme dans AlphaGo).

---

**Q : Pourquoi state_size = 78 et pas juste 75 (25+25+25 pour les pièces) ?**

R : 
- 25+25+25 = 75 = positions des 3 types de pièces sur le plateau 5×5
- +1 = indice 75 : joueur courant. Nécessaire car l'agent joue les deux rôles dans l'environnement (acteur et côté adverse géré par le wrapper)
- +1 = indice 76 : phase courante (0=bouger Bobail, 1=bouger pion). Sans ça, le même état de plateau aurait deux significations différentes selon la phase
- +1 = indice 77 : premier tour. Au premier tour de J0, les règles sont différentes (pas de déplacement du Bobail). Sans ce bit, l'agent ne peut pas distinguer "début de partie" de "milieu de partie en phase 1"

---

## Questions sur l'implémentation

**Q : Que contient un fichier .pt ?**

R : Un fichier `.pt` (PyTorch) sauvegardé avec `torch.save()` contient un dictionnaire Python sérialisé. Dans notre cas :
- Pour REINFORCE/MB : `{"policy": state_dict, "lr": 0.001, "gamma": 0.99, "hidden_sizes": [128,128]}`
- Pour REINFORCE Critic : `{"actor": state_dict, "critic": state_dict, "lr": 0.001, "gamma": 0.99, "value_coef": 0.5, "hidden_sizes": [128,128]}`

Le `state_dict` est un OrderedDict PyTorch contenant tous les tenseurs de poids et biais du réseau.

---

**Q : Pourquoi le modèle Bobail Critic est incompatible avec l'ancien environnement ?**

R : La couche d'entrée du réseau est `nn.Linear(state_size, 128)`. Si state_size=77 à l'entraînement, la matrice de poids est de taille (128 × 77). Si on essaie de charger ces poids dans un réseau avec state_size=78 → matrice (128 × 78) → incompatibilité de dimensions → PyTorch lève une erreur.

---

**Q : C'est quoi TensorBoard et pourquoi on l'utilise ?**

R : TensorBoard est un outil de visualisation développé par Google (intégré à PyTorch). Pendant l'entraînement, on écrit des scalaires avec `writer.add_scalar("train/loss", valeur, episode)`. TensorBoard affiche ensuite des courbes d'apprentissage en temps réel dans un navigateur. On logue : score moyen d'entraînement, longueur des épisodes, valeur de la perte. Ça permet de détecter les problèmes (divergence, stagnation) pendant l'entraînement.

---

# PARTIE 8 — COMMENT PRÉSENTER TA PARTIE À L'ORAL

## Structure suggérée (5 minutes)

**1. Introduction (30 sec)**
> "Ma partie porte sur REINFORCE et ses deux variantes : REINFORCE avec baseline moyenne et REINFORCE avec réseau critique. Ces trois algorithmes font partie de la famille des méthodes de gradient de politique, qui optimisent directement une politique stochastique."

**2. L'algorithme de base (1 min)**
> "REINFORCE, proposé par Williams en 1992, fonctionne en épisodes complets. L'agent joue une partie entière, calcule les retours cumulés G_t pour chaque pas, puis met à jour la politique en renforçant les actions qui ont mené à de bons résultats. La perte est -Σ log π(a_t|s_t) · G_t."

**3. Le problème et les solutions (1.5 min)**
> "Le problème principal est la haute variance des gradients. J'ai implémenté deux solutions : la baseline moyenne, qui soustrait la moyenne des retours de l'épisode pour centrer les avantages, et le réseau critique, qui apprend une fonction de valeur V(s) pour estimer les avantages de façon adaptive."

**4. Les résultats (1.5 min)**
> "Sur les environnements simples LineWorld et GridWorld, les trois variantes convergent vers des performances identiques. Sur TicTacToe, la hiérarchie théorique est validée : REINFORCE Critic atteint 0.987, Mean Baseline 0.939, REINFORCE pur 0.915. Sur Bobail, j'ai dû ré-entraîner le meilleur modèle suite à une correction des règles du premier tour."

**5. Conclusion (30 sec)**
> "Ces résultats confirment que la réduction de variance est un facteur clé dans les environnements complexes. Plus la baseline est précise, meilleure est la convergence."

---

## Phrases à avoir en tête

- "REINFORCE est une méthode Monte-Carlo car elle utilise l'épisode complet pour estimer le gradient."
- "La baseline ne biaise pas le gradient car son espérance par rapport à la politique est nulle."
- "Le .detach() est indispensable pour que les deux réseaux aient des objectifs indépendants."
- "Le masquage des actions illégales garantit que l'agent ne joue jamais de coup invalide."
- "L'encodage de l'état à 78 dimensions permet au réseau de distinguer la phase du jeu et le premier tour."

---

## Si tu ne sais pas répondre

Ne panique pas. Tu peux dire :
- "Je ne me souviens pas de la formule exacte mais le principe est..."
- "Dans mon implémentation, j'ai fait le choix de... parce que..."
- "D'après mes résultats expérimentaux, on observe que..."

Le prof veut voir que tu comprends les concepts, pas que tu as mémorisé des formules.

---

# PARTIE 9 — GLOSSAIRE COMPLET

| Terme | Définition simple |
|-------|------------------|
| **Politique π** | Stratégie de l'agent : probabilité de chaque action dans chaque situation |
| **Retour G_t** | Somme des récompenses futures depuis le pas t, actualisées par γ |
| **Gradient de politique** | Direction dans laquelle modifier les poids pour améliorer la politique |
| **Monte-Carlo** | Estimation basée sur un épisode complet (vs TD qui met à jour à chaque pas) |
| **Baseline** | Valeur soustraite au retour pour réduire la variance sans biaiser |
| **Variance** | Variabilité des mises à jour d'un épisode à l'autre |
| **Biais** | Erreur systématique dans l'estimation du gradient |
| **Logit** | Nombre brut avant softmax (pas encore une probabilité) |
| **Softmax** | Transforme des logits en probabilités qui somment à 1 |
| **ReLU** | Fonction d'activation : max(0, x) |
| **Adam** | Optimiseur adaptatif (plus intelligent que descente de gradient basique) |
| **Gradient clipping** | Plafonnement de la norme du gradient pour éviter les explosions |
| **Acteur** | Le réseau de politique π_θ (choisit les actions) |
| **Critique** | Le réseau de valeur V(s;w) (évalue les états) |
| **Avantage A_t** | G_t - V(s_t) : mesure si un coup était meilleur que prévu |
| **.detach()** | Coupe le lien de gradient entre deux calculs |
| **state_dict** | Dictionnaire PyTorch contenant tous les poids d'un réseau |
| **Checkpoint** | Sauvegarde du modèle à un moment donné (1K, 10K, 100K épisodes) |
| **Greedy** | Mode d'action qui choisit toujours l'action la plus probable (pas de hasard) |
| **Stochastique** | Mode d'action qui tire au hasard selon les probabilités (exploration) |
| **γ (gamma)** | Facteur d'actualisation : importance des récompenses futures |
| **α (alpha)** | Taux d'apprentissage : amplitude des mises à jour |
| **value_coef** | Coefficient qui équilibre la perte acteur et la perte critique |
| **TensorBoard** | Outil de visualisation des courbes d'apprentissage |
| **self-play** | Entraînement où l'agent joue contre lui-même |
| **BobailVsRandom** | Wrapper qui fait jouer un adversaire aléatoire automatiquement |
