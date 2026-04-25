import numpy as np
import random
import time
import copy
from collections import deque
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from envs.base_env import BaseEnv


# ─────────────────────────────────────────────────────────────────────────────
# Réseau de neurones
# ─────────────────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    Réseau fully-connected : état → Q(s, a) pour toutes les actions.

    Architecture par défaut : state_size → 128 → 128 → action_size
    """

    def __init__(self, state_size: int, action_size: int, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        in_size = state_size
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.ReLU()]
            in_size = h
        layers.append(nn.Linear(in_size, action_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Agent Deep Q-Learning
# ─────────────────────────────────────────────────────────────────────────────

class DeepQLearning:
    """
    Agent Deep Q-Learning (DQN) avec target network.

    ALGORITHME (Mnih et al., 2013 — arXiv:1312.5602)
    ──────────────────────────────────────────────────
    A chaque step :
        1. Choisir une action avec politique epsilon-greedy
        2. Executer l'action -> obtenir (s', r, done)
        3. Stocker la transition dans le replay buffer (deque)
        4. Tirer un mini-batch du replay buffer
        5. Calculer les cibles via le TARGET NETWORK :
               y = r                                       si done
               y = r + gamma * max_a' Q_target(s', a')    sinon
        6. Mise à jour de Q_online par descente de gradient (MSE loss)
        7. Toutes les `target_update_freq` steps : copier Q_online → Q_target
        8. Décrémenter epsilon

    HYPERPARAMÈTRES
    ───────────────
        lr                 : taux d'apprentissage Adam          défaut 1e-3
        gamma              : facteur de discount                défaut 0.99
        epsilon            : exploration initiale               défaut 1.0
        epsilon_min        : exploration minimale               défaut 0.01
        epsilon_decay      : décroissance par épisode           défaut 0.995
        batch_size         : taille du mini-batch               défaut 64
        buffer_size        : taille max du replay buffer        défaut 10_000
        hidden_sizes       : architecture du réseau             défaut (128, 128)
        target_update_freq : fréquence de sync du target net    défaut 1_000  # [TARGET NETWORK]
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
        buffer_size: int = 10_000,
        hidden_sizes: tuple = (128, 128),
        target_update_freq: int = 1_000,   # [TARGET NETWORK]
    ):
        self.lr                 = lr
        self.gamma              = gamma
        self.epsilon            = epsilon
        self.epsilon_min        = epsilon_min
        self.epsilon_decay      = epsilon_decay
        self.batch_size         = batch_size
        self.buffer_size        = buffer_size
        self.hidden_sizes       = hidden_sizes
        self.target_update_freq = target_update_freq  # [TARGET NETWORK]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Réseau online (mis à jour à chaque step)
        self._q_net: Optional[QNetwork] = None
        # Réseau cible (mis à jour toutes les C steps)     # [TARGET NETWORK]
        self._target_net: Optional[QNetwork] = None        # [TARGET NETWORK]

        self._optimizer = None
        self._criterion = nn.MSELoss()

        # Replay buffer
        self._buffer: deque = deque(maxlen=buffer_size)

        # Compteur global de steps pour la sync du target net  # [TARGET NETWORK]
        self._total_steps: int = 0                             # [TARGET NETWORK]

    # ── Construction paresseuse des réseaux ──────────────────────────────────

    def _ensure_networks(self, state_size: int, action_size: int) -> None:
        if self._q_net is not None:
            return
        self._q_net = QNetwork(state_size, action_size, self.hidden_sizes).to(self.device)
        # [TARGET NETWORK] Copie initiale : target_net = q_net
        self._target_net = copy.deepcopy(self._q_net).to(self.device)   # [TARGET NETWORK]
        self._target_net.eval()                                           # [TARGET NETWORK]
        self._optimizer = optim.Adam(self._q_net.parameters(), lr=self.lr)

    # [TARGET NETWORK] Synchronisation hard-copy : θ_target ← θ_online
    def _update_target_network(self) -> None:                             # [TARGET NETWORK]
        self._target_net.load_state_dict(self._q_net.state_dict())        # [TARGET NETWORK]

    # ── Fabrique depuis config dict ───────────────────────────────────────────

    @classmethod
    def from_config(cls, config: dict) -> "DeepQLearning":
        return cls(
            lr                 = float(config.get("lr",                 1e-3)),
            gamma              = float(config.get("gamma",              0.99)),
            epsilon            = float(config.get("epsilon",            1.0)),
            epsilon_min        = float(config.get("epsilon_min",        0.01)),
            epsilon_decay      = float(config.get("epsilon_decay",      0.995)),
            batch_size         = int(config.get("batch_size",           64)),
            buffer_size        = int(config.get("buffer_size",          10_000)),
            hidden_sizes       = tuple(config.get("hidden_sizes",       [128, 128])),
            target_update_freq = int(config.get("target_update_freq",   1_000)),
        )

    # ── Interface principale ──────────────────────────────────────────────────

    def select_action(self, env: BaseEnv, greedy: bool = False) -> int:
        """
        Politique epsilon-greedy.
        greedy=True → exploitation pure (évaluation et démo GUI).
        """
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
        log_dir: str = "runs/deep_q_learning",
        log_every: int = 500,
        save_callback=None,
    ) -> dict:
        """
        Entraîne l'agent, logue dans TensorBoard et évalue aux checkpoints.

        TensorBoard :
            train/mean_score          : score moyen glissant pendant l'entraînement
            train/mean_length         : longueur moyenne des parties
            train/epsilon             : valeur courante de epsilon
            train/mean_loss           : perte MSE moyenne (courbe de loss)
            train/target_updates      : nombre de syncs du target network       # [TARGET NETWORK]
            eval/mean_score           : score greedy au checkpoint
            eval/mean_length          : longueur greedy au checkpoint
            eval/mean_action_time_ms

        Args:
            save_callback : signature save_callback(agent, checkpoint, is_best)

        Returns:
            dict : métriques aux checkpoints
        """
        self._ensure_networks(env.state_size, env.action_size)

        writer             = SummaryWriter(log_dir=log_dir)
        checkpoint_results = {}
        checkpoints_done   = set()
        episode_scores     = []
        episode_lengths    = []
        losses             = []
        best_score         = -float("inf")
        target_updates     = 0   # [TARGET NETWORK]

        for episode in range(1, n_episodes + 1):
            # ── Épisode d'entraînement ─────────────────────────────────────
            state     = env.reset()
            ep_score  = 0.0
            ep_steps  = 0

            while not env.is_game_over():
                action = self.select_action(env)
                next_state, reward, done = env.step(action)

                # Stocker la transition
                self._buffer.append((state, action, reward, next_state, done))
                state = next_state

                self._total_steps += 1  # [TARGET NETWORK]

                # Apprentissage si le buffer est suffisamment rempli
                if len(self._buffer) >= self.batch_size:
                    loss = self._learn(env.action_size)
                    losses.append(loss)

                # [TARGET NETWORK] Sync hard-copy toutes les C steps
                if self._total_steps % self.target_update_freq == 0:  # [TARGET NETWORK]
                    self._update_target_network()                       # [TARGET NETWORK]
                    target_updates += 1                                 # [TARGET NETWORK]

                ep_score += reward
                ep_steps += 1
                #if ep_steps % 100 == 0:
                   # print("ep_steps =", ep_steps, "is_game_over =", env.is_game_over())

            self._decay_epsilon()
            episode_scores.append(ep_score)
            episode_lengths.append(ep_steps)

            # ── Log TensorBoard ────────────────────────────────────────────
            if episode % log_every == 0:
                mean_score  = np.mean(episode_scores[-log_every:])
                mean_length = np.mean(episode_lengths[-log_every:])
                mean_loss   = np.mean(losses[-log_every * 10:]) if losses else 0.0

                writer.add_scalar("train/mean_score",     mean_score,    episode)
                writer.add_scalar("train/mean_length",    mean_length,   episode)
                writer.add_scalar("train/epsilon",        self.epsilon,  episode)
                writer.add_scalar("train/mean_loss",      mean_loss,     episode)
                writer.add_scalar("train/target_updates", target_updates, episode)  # [TARGET NETWORK]

            # ── Évaluation aux checkpoints ─────────────────────────────────
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
                      f"target_updates={target_updates}"           # [TARGET NETWORK]
                      f"{' ← BEST' if is_best else ''}")

        writer.close()
        return checkpoint_results

    def evaluate(self, env: BaseEnv, n_episodes: int = 500) -> dict:
        """Évalue la policy en mode GREEDY (epsilon=0)."""
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

        mean_score = total_score / n_episodes if n_episodes > 0 else 0.0
        mean_length = total_steps / n_episodes if n_episodes > 0 else 0.0
        mean_action_time_ms = (total_time / total_steps) * 1000 if total_steps > 0 else 0.0

        return {
            "mean_score":          mean_score,
            "mean_length":         mean_length,
            "mean_action_time_ms": mean_action_time_ms,
        }

        """return {
            "mean_score":          total_score / n_episodes,
            "mean_length":         total_steps / n_episodes,
            "mean_action_time_ms": (total_time / total_steps) * 1000 if total_steps > 0 else 0,
        }"""

    # ── Méthodes privées ──────────────────────────────────────────────────────

    def _learn(self, action_size: int) -> float:
        """
        Tire un mini-batch et effectue une mise à jour du réseau online.

        Cibles Bellman via le TARGET NETWORK (Mnih et al., 2013) :  # [TARGET NETWORK]
            y = r                                       si done
            y = r + gamma * max_a' Q_target(s', a')    sinon        # [TARGET NETWORK]
        """
        batch = random.sample(self._buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t      = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # Q_online(s, a) — réseau entraîné
        q_current = self._q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Cible : r + gamma * max Q_target(s', .) si non terminal  # [TARGET NETWORK]
        with torch.no_grad():
            q_next  = self._target_net(next_states_t).max(1)[0]    # [TARGET NETWORK]
            targets = rewards_t + self.gamma * q_next * (1 - dones_t)

        loss = self._criterion(q_current, targets)

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._q_net.parameters(), max_norm=1.0) #Pour éliminer les loss explosives
        self._optimizer.step()

        return loss.item()

    def _decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Sérialisation ─────────────────────────────────────────────────────────

    def get_state_dict(self) -> dict:
        return {
            "q_net":              self._q_net.state_dict() if self._q_net else None,
            "target_net":         self._target_net.state_dict() if self._target_net else None,  # [TARGET NETWORK]
            "lr":                 self.lr,
            "gamma":              self.gamma,
            "epsilon":            self.epsilon,
            "epsilon_min":        self.epsilon_min,
            "epsilon_decay":      self.epsilon_decay,
            "batch_size":         self.batch_size,
            "buffer_size":        self.buffer_size,
            "hidden_sizes":       self.hidden_sizes,
            "target_update_freq": self.target_update_freq,  # [TARGET NETWORK]
            "total_steps":        self._total_steps,        # [TARGET NETWORK]
        }

    def load_state_dict(self, data: dict, state_size: int, action_size: int) -> None:
        self.lr                 = data["lr"]
        self.gamma              = data["gamma"]
        self.epsilon            = data["epsilon"]
        self.epsilon_min        = data["epsilon_min"]
        self.epsilon_decay      = data["epsilon_decay"]
        self.batch_size         = data["batch_size"]
        self.buffer_size        = data["buffer_size"]
        self.hidden_sizes       = data["hidden_sizes"]
        self.target_update_freq = data.get("target_update_freq", 1_000)  # [TARGET NETWORK]
        self._total_steps       = data.get("total_steps", 0)             # [TARGET NETWORK]
        self._ensure_networks(state_size, action_size)
        if data["q_net"] is not None:
            self._q_net.load_state_dict(data["q_net"])
        if data.get("target_net") is not None:                            # [TARGET NETWORK]
            self._target_net.load_state_dict(data["target_net"])          # [TARGET NETWORK]