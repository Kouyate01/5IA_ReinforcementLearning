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
from agents.dqn import QNetwork


class DoubleDeepQLearning:
    """
    Agent Double Deep Q-Learning (DDQN).

    ALGORITHME
    ──────────
    Identique à DQN mais avec deux réseaux :
        - Q-net    (online)  : mis à jour à chaque step par gradient
        - Q-target (frozen)  : copie périodique du Q-net (tous les
                               `target_update_freq` épisodes)

    Calcul de la cible :
        a* = argmax_a Q_online(s', a)          ← sélection par le réseau online
        y  = r + gamma * Q_target(s', a*)      ← évaluation par le réseau cible

    → Découple sélection et évaluation → réduit la surestimation des Q-valeurs

    HYPERPARAMÈTRES SUPPLÉMENTAIRES (vs DQN)
    ─────────────────────────────────────────
        target_update_freq : fréquence de mise à jour du réseau cible  défaut 10
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
        target_update_freq: int = 10,
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._q_net: Optional[QNetwork]    = None
        self._q_target: Optional[QNetwork] = None
        self._optimizer = None
        self._criterion = nn.MSELoss()

        self._buffer: deque = deque(maxlen=buffer_size)
        self._episode_count: int = 0

    def _ensure_networks(self, state_size: int, action_size: int) -> None:
        if self._q_net is not None:
            return
        self._q_net    = QNetwork(state_size, action_size, self.hidden_sizes).to(self.device)
        self._q_target = QNetwork(state_size, action_size, self.hidden_sizes).to(self.device)
        # Initialiser le réseau cible avec les mêmes poids
        self._q_target.load_state_dict(self._q_net.state_dict())
        self._q_target.eval()   # jamais en mode training
        self._optimizer = optim.Adam(self._q_net.parameters(), lr=self.lr)

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
        log_dir: str = "runs/double_deep_q_learning",
        log_every: int = 500,
        save_callback=None,
    ) -> dict:
        self._ensure_networks(env.state_size, env.action_size)

        writer             = SummaryWriter(log_dir=log_dir)
        checkpoint_results = {}
        checkpoints_done   = set()
        episode_scores     = []
        episode_lengths    = []
        losses             = []
        best_score         = -float("inf")

        for episode in range(1, n_episodes + 1):
            state     = env.reset()
            ep_score  = 0.0
            ep_steps  = 0

            while not env.is_game_over():
                action = self.select_action(env)
                next_state, reward, done = env.step(action)

                self._buffer.append((state, action, reward, next_state, done))
                state = next_state

                if len(self._buffer) >= self.batch_size:
                    loss = self._learn(env.action_size)
                    losses.append(loss)

                ep_score += reward
                ep_steps += 1

            self._decay_epsilon()
            self._episode_count += 1
            episode_scores.append(ep_score)
            episode_lengths.append(ep_steps)

            # Mise à jour périodique du réseau cible
            if self._episode_count % self.target_update_freq == 0:
                self._q_target.load_state_dict(self._q_net.state_dict())

            if episode % log_every == 0:
                mean_score  = np.mean(episode_scores[-log_every:])
                mean_length = np.mean(episode_lengths[-log_every:])
                mean_loss   = np.mean(losses[-log_every * 10:]) if losses else 0.0

                writer.add_scalar("train/mean_score",  mean_score,   episode)
                writer.add_scalar("train/mean_length", mean_length,  episode)
                writer.add_scalar("train/epsilon",     self.epsilon, episode)
                writer.add_scalar("train/mean_loss",   mean_loss,    episode)

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
                      f"loss={np.mean(losses[-500:]):.4f}"
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

    def _learn(self, action_size: int) -> float:
        """
        Mise à jour Double DQN.

        Sélection de la meilleure action via Q_online :
            a* = argmax_a Q_online(s', a)

        Évaluation de cette action via Q_target :
            y = r + gamma * Q_target(s', a*)
        """
        batch = random.sample(self._buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t      = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # Q(s, a) courant
        q_current = self._q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Sélection : argmax sur le réseau online
            best_actions = self._q_net(next_states_t).argmax(1)
            # Évaluation : Q-valeur du réseau cible pour cette action
            q_next = self._q_target(next_states_t).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            targets = rewards_t + self.gamma * q_next * (1 - dones_t)

        loss = self._criterion(q_current, targets)

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._q_net.parameters(), max_norm=1.0) #gradient clipping pour éviter qu'un lr élevé (1e-3) fasse exploser les gradients dès les premiers batchs
        self._optimizer.step()

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
        self.target_update_freq = data["target_update_freq"]
        self._ensure_networks(state_size, action_size)
        if data["q_net"] is not None:
            self._q_net.load_state_dict(data["q_net"])
        if data["q_target"] is not None:
            self._q_target.load_state_dict(data["q_target"])