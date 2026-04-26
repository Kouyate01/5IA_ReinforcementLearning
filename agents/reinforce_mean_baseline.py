import numpy as np
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from envs.base_env import BaseEnv


class PolicyNetwork(nn.Module):
    """Réseau fully-connected : état → logits sur toutes les actions."""

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


class REINFORCEMeanBaseline:
    """
    REINFORCE avec baseline constante = moyenne des retours de l'épisode.

    ALGORITHME
    ──────────
    Identique à REINFORCE mais l'avantage utilisé pour le gradient est :
        Â_t = G_t − mean(G)   avec  mean(G) = (1/T) Σ_t G_t

    Soustaire la moyenne ne biaise pas l'estimateur du gradient
    (car mean(G) est constant vis-à-vis de θ) mais réduit la variance :
    les coups d'un épisode « moyen » reçoivent un avantage nul,
    les coups d'un bon épisode un avantage positif, et inversement.

    HYPERPARAMÈTRES
    ───────────────
        lr           : taux d'apprentissage Adam   défaut 1e-3
        gamma        : facteur de discount          défaut 0.99
        hidden_sizes : architecture du réseau       défaut (128, 128)
    """

    CHECKPOINTS = [1_000, 10_000, 100_000, 1_000_000]

    def __init__(
        self,
        lr: float = 1e-3,
        gamma: float = 0.99,
        hidden_sizes: tuple = (128, 128),
    ):
        self.lr           = lr
        self.gamma        = gamma
        self.hidden_sizes = hidden_sizes
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._policy: Optional[PolicyNetwork] = None
        self._optimizer = None

    def _ensure_network(self, state_size: int, action_size: int) -> None:
        if self._policy is not None:
            return
        self._policy    = PolicyNetwork(state_size, action_size, self.hidden_sizes).to(self.device)
        self._optimizer = optim.Adam(self._policy.parameters(), lr=self.lr)

    @classmethod
    def from_config(cls, config: dict) -> "REINFORCEMeanBaseline":
        return cls(
            lr           = float(config.get("lr",           1e-3)),
            gamma        = float(config.get("gamma",        0.99)),
            hidden_sizes = tuple(config.get("hidden_sizes", [128, 128])),
        )

    # ── Interface principale ─────────────────────────────────────────────────

    def select_action(self, env: BaseEnv, greedy: bool = False) -> int:
        """
        Échantillonne une action selon π_θ avec masquage des coups illégaux.
        greedy=True → argmax des probabilités (évaluation et démo GUI).
        """
        self._ensure_network(env.state_size, env.action_size)
        actions = env.available_actions()

        state_t = torch.FloatTensor(env.get_state()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._policy(state_t).squeeze(0)

        mask = torch.full((env.action_size,), float("-inf"), device=self.device)
        for a in actions:
            mask[a] = 0.0
        probs = F.softmax(logits + mask, dim=-1)

        if greedy:
            return int(probs.argmax().item())
        return int(Categorical(probs).sample().item())

    def train(
        self,
        env: BaseEnv,
        n_episodes: int = 100_000,
        eval_episodes: int = 500,
        log_dir: str = "runs/reinforce_mean_baseline",
        log_every: int = 500,
        save_callback=None,
    ) -> dict:
        """
        Entraîne l'agent, logue dans TensorBoard et évalue aux checkpoints.

        TensorBoard :
            train/mean_score, train/mean_length, train/mean_loss
            eval/mean_score, eval/mean_length, eval/mean_action_time_ms

        Returns:
            dict : métriques aux checkpoints
        """
        self._ensure_network(env.state_size, env.action_size)

        writer             = SummaryWriter(log_dir=log_dir)
        checkpoint_results = {}
        checkpoints_done   = set()
        episode_scores     = []
        episode_lengths    = []
        policy_losses      = []
        best_score         = -float("inf")

        for episode in range(1, n_episodes + 1):
            log_probs, rewards = self._collect_episode(env)

            ep_score = sum(rewards)
            ep_steps = len(rewards)
            episode_scores.append(ep_score)
            episode_lengths.append(ep_steps)

            returns = self._compute_returns(rewards)
            loss    = self._update_policy(log_probs, returns)
            policy_losses.append(loss)

            if episode % log_every == 0:
                mean_score  = np.mean(episode_scores[-log_every:])
                mean_length = np.mean(episode_lengths[-log_every:])
                mean_loss   = np.mean(policy_losses[-log_every:])

                writer.add_scalar("train/mean_score",  mean_score,  episode)
                writer.add_scalar("train/mean_length", mean_length, episode)
                writer.add_scalar("train/mean_loss",   mean_loss,   episode)

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
                      f"loss={np.mean(policy_losses[-500:]):.4f}"
                      f"{' ← BEST' if is_best else ''}")

        writer.close()
        return checkpoint_results

    def evaluate(self, env: BaseEnv, n_episodes: int = 500) -> dict:
        """Évalue la politique en mode greedy (argmax des probs)."""
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
            "mean_action_time_ms": (total_time / total_steps) * 1000 if total_steps > 0 else 0.0,
        }

    # ── Méthodes privées ─────────────────────────────────────────────────────

    def _collect_episode(self, env: BaseEnv):
        """Déroule un épisode complet en suivant π_θ."""
        state     = env.reset()
        log_probs = []
        rewards   = []

        while not env.is_game_over():
            actions = env.available_actions()
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits  = self._policy(state_t).squeeze(0)

            mask = torch.full((env.action_size,), float("-inf"), device=self.device)
            for a in actions:
                mask[a] = 0.0
            log_probs_all = F.log_softmax(logits + mask, dim=-1)

            action_t = Categorical(log_probs_all.exp()).sample()
            log_probs.append(log_probs_all[action_t])

            state, reward, _ = env.step(action_t.item())
            rewards.append(reward)

        return log_probs, rewards

    def _compute_returns(self, rewards: list) -> list:
        """Retours Monte-Carlo discountés G_t = Σ_{k≥t} γ^(k-t) · r_k."""
        G       = 0.0
        returns = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns

    def _update_policy(self, log_probs: list, returns: list) -> float:
        """
        Mise à jour avec avantage A_t = G_t − mean(G).

        La baseline mean(G) est constante par rapport à θ → pas de biais.
        Elle centre les retours autour de 0, ce qui réduit la variance
        du gradient estimé comparé au REINFORCE sans baseline.
        """
        returns_t = torch.FloatTensor(returns).to(self.device)
        baseline  = returns_t.mean()
        advantages = returns_t - baseline

        loss = torch.stack([-lp * adv for lp, adv in zip(log_probs, advantages)]).sum()

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._policy.parameters(), max_norm=1.0)
        self._optimizer.step()

        return loss.item()

    # ── Sérialisation ─────────────────────────────────────────────────────────

    def get_state_dict(self) -> dict:
        return {
            "policy":       self._policy.state_dict() if self._policy else None,
            "lr":           self.lr,
            "gamma":        self.gamma,
            "hidden_sizes": self.hidden_sizes,
        }

    def load_state_dict(self, data: dict, state_size: int, action_size: int) -> None:
        self.lr           = data["lr"]
        self.gamma        = data["gamma"]
        self.hidden_sizes = tuple(data["hidden_sizes"])
        self._ensure_network(state_size, action_size)
        if data["policy"] is not None:
            self._policy.load_state_dict(data["policy"])
