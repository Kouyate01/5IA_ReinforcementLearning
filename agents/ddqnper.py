import numpy as np
import random
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from envs.base_env import BaseEnv
from agents.dqn import QNetwork


# ─────────────────────────────────────────────────────────────────────────────
# Replay Buffer Prioritisé (Sum-Tree)
# ─────────────────────────────────────────────────────────────────────────────

class SumTree:
    """
    Arbre binaire de sommes permettant un échantillonnage en O(log N)
    proportionnel aux priorités.

    Structure :
        - Les feuilles (indices [n-1 .. 2n-2]) stockent les priorités p_i
        - Les noeuds internes stockent la somme de leurs enfants
        - La racine contient la somme totale Σ p_i

    Exemple pour N=4 :
                 [Σ]
               /      \
          [p0+p1]  [p2+p3]
          /    \    /    \
        [p0] [p1] [p2] [p3]

    Échantillonnage : on tire v ~ Uniform(0, total) et on descend l'arbre
    en allant à gauche si v <= fils_gauche, à droite sinon.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data     = np.empty(capacity, dtype=object)
        self._write   = 0   # prochain indice d'écriture (circulaire)
        self._n_entries = 0

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def _propagate(self, idx: int, change: float) -> None:
        """Remonte la mise à jour de priorité jusqu'à la racine."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Descend l'arbre pour trouver la feuille correspondant à la valeur s."""
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def add(self, priority: float, data) -> None:
        """Ajoute une transition avec la priorité donnée."""
        leaf_idx = self._write + self.capacity - 1
        self.data[self._write] = data
        self.update(leaf_idx, priority)
        self._write = (self._write + 1) % self.capacity
        self._n_entries = min(self._n_entries + 1, self.capacity)

    def update(self, leaf_idx: int, priority: float) -> None:
        """Met à jour la priorité d'une feuille existante."""
        change = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, change)

    def get(self, s: float):
        """
        Retourne (leaf_idx, priority, data) pour la valeur s tirée uniformément
        dans [0, total].
        """
        leaf_idx  = self._retrieve(0, s)
        data_idx  = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def __len__(self) -> int:
        return self._n_entries


class PrioritizedReplayBuffer:
    """
    Replay buffer avec échantillonnage prioritisé (Schaul et al., 2015).

    IDÉE CENTRALE
    ─────────────
    Au lieu de tirer les transitions uniformément, on tire avec une probabilité
    proportionnelle à |TD-error|^alpha :
        p_i = (|δ_i| + ε)^α

    Où :
        δ_i = erreur TD de la transition i
        ε   = petite constante pour éviter p_i = 0           défaut 1e-6
        α   = contrôle la force de la priorisation (0=uniforme) défaut 0.6

    CORRECTION DU BIAIS (Importance Sampling)
    ───────────────────────────────────────────
    L'échantillonnage non-uniforme introduit un biais.
    On le corrige en pondérant chaque gradient par :
        w_i = (1 / (N * p_i))^β      (normalisés par max w_i)

    β commence faible et monte vers 1 au cours de l'entraînement
    (on corrige de moins en moins au début, de plus en plus à la fin).

    PARAMÈTRES
    ──────────
        alpha    : force de la priorisation        défaut 0.6
        beta     : force de la correction IS       défaut 0.4
        beta_max : valeur finale de beta           défaut 1.0
        eps      : constante de stabilité          défaut 1e-6
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float  = 0.4,
        beta_max: float = 1.0,
        eps: float   = 1e-6,
    ):
        self.capacity  = capacity
        self.alpha     = alpha
        self.beta      = beta
        self.beta_init = beta # <- pour stocker la valeur initiale
        self.beta_max  = beta_max
        self.eps       = eps
        self._tree     = SumTree(capacity)

        # Priorité maximale observée (utilisée pour les nouvelles transitions)
        self._max_priority: float = 1.0

    def add(self, transition) -> None:
        """Ajoute une transition avec la priorité maximale courante."""
        priority = self._max_priority ** self.alpha
        self._tree.add(priority, transition)

    def sample(self, batch_size: int):
        """
        Échantillonne `batch_size` transitions.

        Retourne :
            indices    : indices des feuilles dans le SumTree (pour mise à jour)
            transitions: liste de transitions
            weights    : poids IS normalisés (torch.FloatTensor)
        """
        indices     = []
        transitions = []
        weights     = []
        segment     = self._tree.total / batch_size
        n           = len(self._tree)

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s  = random.uniform(lo, hi)

            leaf_idx, priority, data = self._tree.get(s)

            # Probabilité de tirage : p_i / Σp
            prob = priority / self._tree.total if self._tree.total > 0 else 1.0 / n
            prob = max(prob, 1e-10)  # évite log(0)

            # Poids IS : (1 / (N * p_i))^β
            w = (1.0 / (n * prob)) ** self.beta

            indices.append(leaf_idx)
            transitions.append(data)
            weights.append(w)

        # Normaliser par le poids maximal du batch
        max_w = max(weights)
        weights = [w / max_w for w in weights]

        return indices, transitions, torch.FloatTensor(weights)

    def update_priorities(self, indices, td_errors: np.ndarray) -> None:
        """Met à jour les priorités après calcul des nouvelles erreurs TD."""
        for idx, td_err in zip(indices, td_errors):
            priority = (abs(td_err) + self.eps) ** self.alpha
            self._tree.update(idx, priority)
            self._max_priority = max(self._max_priority, priority)

    def anneal_beta(self, step: int, total_steps: int) -> None:
        """Augmente beta linéairement de beta_init → beta_max."""
        fraction   = min(1.0, step / total_steps)
        self.beta  = self.beta_init + fraction * (self.beta_max - self.beta_init)

    def __len__(self) -> int:
        return len(self._tree)


# ─────────────────────────────────────────────────────────────────────────────
# Agent DDQN + PER
# ─────────────────────────────────────────────────────────────────────────────

class DoubleDeepQLearningWithPrioritizedExperienceReplay:
    """
    Agent Double DQN avec Prioritized Experience Replay (DDQN + PER).

    ALGORITHME
    ──────────
    Même base que DDQN + ER, mais le replay buffer :
        - Trie les transitions par priorité  (|TD-error|^α)
        - Échantillonne proportionnellement à cette priorité
        - Corrige le biais introduit par des poids IS (β → 1)

    La mise à jour du gradient devient :
        Loss = Σ w_i * (Q(s_i, a_i) - y_i)²

    Les priorités sont mises à jour après chaque batch avec les nouveaux |δ_i|.

    HYPERPARAMÈTRES SUPPLÉMENTAIRES
    ─────────────────────────────────
        per_alpha    : force de la priorisation  (0=uniforme)  défaut 0.6
        per_beta     : correction IS initiale                  défaut 0.4
        per_beta_max : correction IS finale (fin entraînement) défaut 1.0
        per_eps      : stabilité numérique                     défaut 1e-6
    """

    CHECKPOINTS = [1_000, 10_000, 100_000, 1_000_000]

    def __init__(
        self,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        hidden_sizes: tuple = (128, 128),
        target_update_freq: int = 10,
        warmup_steps: int = 1_000,
        per_alpha: float = 0.6,
        per_beta: float  = 0.4,
        per_beta_max: float = 1.0,
        per_eps: float   = 1e-6,
    ):
        self.lr                 = lr
        self.gamma              = gamma
        self.epsilon            = epsilon
        self.epsilon_min        = epsilon_min
        self.epsilon_decay      = epsilon_decay
        self.batch_size         = batch_size
        self.buffer_size        = buffer_size
        self.hidden_sizes       = hidden_sizes
        self.target_update_freq = target_update_freq
        self.warmup_steps       = warmup_steps
        self.per_alpha          = per_alpha
        self.per_beta           = per_beta
        self.per_beta_max       = per_beta_max
        self.per_eps            = per_eps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._q_net: Optional[QNetwork]    = None
        self._q_target: Optional[QNetwork] = None
        self._optimizer = None
        # Pas de MSELoss ici : on fait la somme pondérée manuellement
        # (les poids IS ont besoin d'entrer dans le calcul de la loss)

        self._buffer: Optional[PrioritizedReplayBuffer] = None
        self._total_steps: int   = 0
        self._episode_count: int = 0

    def _ensure_networks(self, state_size: int, action_size: int) -> None:
        if self._q_net is not None:
            return
        self._q_net    = QNetwork(state_size, action_size, self.hidden_sizes).to(self.device)
        self._q_target = QNetwork(state_size, action_size, self.hidden_sizes).to(self.device)
        self._q_target.load_state_dict(self._q_net.state_dict())
        self._q_target.eval()
        self._optimizer = optim.Adam(self._q_net.parameters(), lr=self.lr)
        self._buffer    = PrioritizedReplayBuffer(
            capacity  = self.buffer_size,
            alpha     = self.per_alpha,
            beta      = self.per_beta,
            beta_max  = self.per_beta_max,
            eps       = self.per_eps,
        )

    @classmethod
    def from_config(cls, config: dict) -> "DoubleDeepQLearningWithPrioritizedExperienceReplay":
        return cls(
            lr                 = float(config.get("lr",                 1e-3)),
            gamma              = float(config.get("gamma",              0.99)),
            epsilon            = float(config.get("epsilon",            1.0)),
            epsilon_min        = float(config.get("epsilon_min",        0.01)),
            epsilon_decay      = float(config.get("epsilon_decay",      0.995)),
            batch_size         = int(config.get("batch_size",         64)),
            buffer_size        = int(config.get("buffer_size",        50000)),
            hidden_sizes       = tuple(config.get("hidden_sizes", [128, 128])),
            target_update_freq = int(config.get("target_update_freq", 10)),
            warmup_steps       = int(config.get("warmup_steps",       1000)),
            per_alpha          = float(config.get("per_alpha",          0.6)),
            per_beta           = float(config.get("per_beta",           0.4)),
            per_beta_max       = float(config.get("per_beta_max",       1.0)),
            per_eps            = float(config.get("per_eps",            1e-6)),
        )

    # ── Interface principale ──────────────────────────────────────────────────

    def select_action(self, env: BaseEnv, greedy: bool = False) -> int:
        actions = env.available_actions()
        self._ensure_networks(env.state_size, env.action_size)

        if not greedy and random.random() < self.epsilon:
            return random.choice(actions)

        state_t = torch.FloatTensor(env.get_state()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self._q_net(state_t).squeeze(0).cpu().numpy()
        return max(actions, key=lambda a: q_values[a])

    def train(
        self,
        env: BaseEnv,
        n_episodes: int = 100_000,
        eval_episodes: int = 500,
        log_dir: str = "runs/ddqn_per",
        log_every: int = 500,
        save_callback=None,
    ) -> dict:
        self._ensure_networks(env.state_size, env.action_size)
        total_train_steps = n_episodes * 50  # estimation pour l'annealing de β

        writer             = SummaryWriter(log_dir=log_dir)
        checkpoint_results = {}
        checkpoints_done   = set()
        episode_scores     = []
        episode_lengths    = []
        losses             = []
        best_score         = -float("inf")

        # ── Warm-up ───────────────────────────────────────────────────────────
        print(f"[Warm-up PER] Collecte de {self.warmup_steps} transitions...")
        self._warmup(env)
        print(f"[Warm-up PER] Terminé — buffer : {len(self._buffer)} transitions")

        for episode in range(1, n_episodes + 1):
            state    = env.reset()
            ep_score = 0.0
            ep_steps = 0

            while not env.is_game_over():
                action = self.select_action(env)
                next_state, reward, done = env.step(action)

                self._buffer.add((state, action, reward, next_state, done))
                state = next_state
                self._total_steps += 1

                # Annealing de β
                self._buffer.anneal_beta(self._total_steps, total_train_steps)

                if len(self._buffer) >= self.batch_size:
                    loss = self._learn(env.action_size)
                    losses.append(loss)

                ep_score += reward
                ep_steps += 1

            self._decay_epsilon()
            self._episode_count += 1
            episode_scores.append(ep_score)
            episode_lengths.append(ep_steps)

            if self._episode_count % self.target_update_freq == 0:
                self._q_target.load_state_dict(self._q_net.state_dict())

            if episode % log_every == 0:
                mean_score  = np.mean(episode_scores[-log_every:])
                mean_length = np.mean(episode_lengths[-log_every:])
                mean_loss   = np.mean(losses[-log_every * 10:]) if losses else 0.0

                writer.add_scalar("train/mean_score",  mean_score,         episode)
                writer.add_scalar("train/mean_length", mean_length,        episode)
                writer.add_scalar("train/epsilon",     self.epsilon,       episode)
                writer.add_scalar("train/mean_loss",   mean_loss,          episode)
                writer.add_scalar("train/per_beta",    self._buffer.beta,  episode)
                writer.add_scalar("train/buffer_size", len(self._buffer),  episode)

            if episode in self.CHECKPOINTS and episode not in checkpoints_done:
                metrics = self.evaluate(env, n_episodes=eval_episodes)
                checkpoint_results[episode] = metrics
                checkpoints_done.add(episode)

                writer.add_scalar("eval/mean_score",          metrics["mean_score"],          episode)
                writer.add_scalar("eval/mean_length",         metrics["mean_length"],         episode)
                writer.add_scalar("eval/mean_action_time_ms", metrics["mean_action_time_ms"], episode)

                is_best = metrics["mean_score"] > best_score
                if is_best:
                    best_score = metrics["mean_score"]

                if save_callback is not None:
                    save_callback(self, episode, is_best)

                print(f"[Checkpoint {episode:>8}] "
                      f"score={metrics['mean_score']:.3f} | "
                      f"length={metrics['mean_length']:.1f} | "
                      f"loss={np.mean(losses[-500:]):.4f} | "
                      f"β={self._buffer.beta:.3f}"
                      f"{' ← BEST' if is_best else ''}")

        writer.close()
        return checkpoint_results

    def evaluate(self, env: BaseEnv, n_episodes: int = 500) -> dict:
        total_score = 0.0
        total_steps = 0
        total_time  = 0.0

        for _ in range(n_episodes):
            env.reset()
            steps = 0
            while not env.is_game_over():
                t0 = time.perf_counter()
                action = self.select_action(env, greedy=True)
                total_time += time.perf_counter() - t0
                env.step(action)
                steps += 1
            total_score += env.score()
            total_steps += steps

        return {
            "mean_score":          total_score / n_episodes,
            "mean_length":         total_steps / n_episodes,
            "mean_action_time_ms": (total_time / total_steps) * 1000 if total_steps > 0 else 0,
        }

    # ── Méthodes privées ──────────────────────────────────────────────────────

    def _warmup(self, env: BaseEnv) -> None:
        steps = 0
        while steps < self.warmup_steps:
            state = env.reset()
            while not env.is_game_over() and steps < self.warmup_steps:
                action = random.choice(env.available_actions())
                next_state, reward, done = env.step(action)
                self._buffer.add((state, action, reward, next_state, done))
                state = next_state
                steps += 1

    def _learn(self, action_size: int) -> float:
        """
        Mise à jour DDQN + PER :
            1. Échantillonnage prioritisé → batch + poids IS
            2. Calcul des cibles Double DQN
            3. Loss pondérée : Σ w_i * δ_i²
            4. Mise à jour des priorités avec |δ_i|
        """
        indices, transitions, weights = self._buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states_t      = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)
        weights_t     = weights.to(self.device)

        q_current = self._q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_actions = self._q_net(next_states_t).argmax(1)
            q_next       = self._q_target(next_states_t).gather(
                1, best_actions.unsqueeze(1)
            ).squeeze(1)
            targets = rewards_t + self.gamma * q_next * (1 - dones_t)

        # Erreurs TD pour mise à jour des priorités
        td_errors = (targets - q_current).detach().cpu().numpy()

        # Loss pondérée par les poids IS
        elementwise_loss = (q_current - targets) ** 2
        loss = (weights_t * elementwise_loss).mean()

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._q_net.parameters(), max_norm=1.0) #gradient clipping pour éviter qu'un lr élevé (1e-3) fasse exploser les gradients dès les premiers batchs
        self._optimizer.step()

        # Mise à jour des priorités dans le SumTree
        self._buffer.update_priorities(indices, td_errors)

        return loss.item()

    def _decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_state_dict(self) -> dict:
        return {
            "q_net":             self._q_net.state_dict() if self._q_net else None,
            "q_target":          self._q_target.state_dict() if self._q_target else None,
            "lr":                self.lr,
            "gamma":             self.gamma,
            "epsilon":           self.epsilon,
            "epsilon_min":       self.epsilon_min,
            "epsilon_decay":     self.epsilon_decay,
            "batch_size":        self.batch_size,
            "buffer_size":       self.buffer_size,
            "hidden_sizes":      self.hidden_sizes,
            "target_update_freq":self.target_update_freq,
            "warmup_steps":      self.warmup_steps,
            "per_alpha":         self.per_alpha,
            "per_beta":          self.per_beta,
            "per_beta_max":      self.per_beta_max,
            "per_eps":           self.per_eps,
        }

    def load_state_dict(self, data: dict, state_size: int, action_size: int) -> None:
        for k, v in data.items():
            if k not in ("q_net", "q_target"):
                setattr(self, k, v)
        self.hidden_sizes = tuple(self.hidden_sizes)
        self._ensure_networks(state_size, action_size)
        if data["q_net"] is not None:
            self._q_net.load_state_dict(data["q_net"])
        if data["q_target"] is not None:
            self._q_target.load_state_dict(data["q_target"])