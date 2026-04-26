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


# ─────────────────────────────────────────────────────────────────────────────
# Réseaux
# ─────────────────────────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """Réseau acteur : état → logits sur toutes les actions."""

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


class ValueNetwork(nn.Module):
    """Réseau critique : état → valeur scalaire V(s)."""

    def __init__(self, state_size: int, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        in_size = state_size
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.ReLU()]
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Agent REINFORCE avec baseline apprise par un critique
# ─────────────────────────────────────────────────────────────────────────────

class REINFORCEWithCritic:
    """
    REINFORCE avec baseline apprise par un réseau critique V(s; w).

    ALGORITHME
    ──────────
    Deux réseaux distincts :
        Acteur  π(a | s ; θ) — politique stochastique
        Critique V(s ; w)   — estimation de la valeur d'état

    À chaque épisode :
        1. Collecter la trajectoire en suivant π_θ ;
           noter les log-probabilités et les valeurs du critique.
        2. Calculer les retours Monte-Carlo G_t.
        3. Calculer l'avantage : A_t = G_t − V(s_t ; w)
        4. Mise à jour :
               Critique : minimiser  Σ_t (G_t − V(s_t ; w))²
               Acteur   : maximiser  Σ_t log π(a_t | s_t ; θ) · A_t.détaché()

    Le .detach() sur les valeurs du critique dans la perte de l'acteur est
    indispensable : on ne veut pas que le gradient de l'acteur remonte
    dans le critique (les deux réseaux ont des objectifs différents).

    HYPERPARAMÈTRES
    ───────────────
        lr           : taux d'apprentissage (acteur et critique)  défaut 1e-3
        gamma        : facteur de discount                         défaut 0.99
        value_coef   : poids de la perte du critique               défaut 0.5
        hidden_sizes : architecture commune des deux réseaux       défaut (128, 128)
    """

    CHECKPOINTS = [1_000, 10_000, 100_000, 1_000_000]

    def __init__(
        self,
        lr: float = 1e-3,
        gamma: float = 0.99,
        value_coef: float = 0.5,
        hidden_sizes: tuple = (128, 128),
    ):
        self.lr           = lr
        self.gamma        = gamma
        self.value_coef   = value_coef
        self.hidden_sizes = hidden_sizes
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._actor:  Optional[PolicyNetwork] = None
        self._critic: Optional[ValueNetwork]  = None
        self._optimizer = None

    # ── Construction paresseuse ──────────────────────────────────────────────

    def _ensure_networks(self, state_size: int, action_size: int) -> None:
        if self._actor is not None:
            return
        self._actor  = PolicyNetwork(state_size, action_size, self.hidden_sizes).to(self.device)
        self._critic = ValueNetwork(state_size, self.hidden_sizes).to(self.device)
        # Un seul optimiseur pour les deux réseaux
        params = list(self._actor.parameters()) + list(self._critic.parameters())
        self._optimizer = optim.Adam(params, lr=self.lr)

    # ── Fabrique depuis config dict ──────────────────────────────────────────

    @classmethod
    def from_config(cls, config: dict) -> "REINFORCEWithCritic":
        return cls(
            lr           = float(config.get("lr",           1e-3)),
            gamma        = float(config.get("gamma",        0.99)),
            value_coef   = float(config.get("value_coef",   0.5)),
            hidden_sizes = tuple(config.get("hidden_sizes", [128, 128])),
        )

    # ── Interface principale ─────────────────────────────────────────────────

    def select_action(self, env: BaseEnv, greedy: bool = False) -> int:
        """
        Échantillonne une action selon π_θ avec masquage des coups illégaux.
        greedy=True → argmax des probabilités (évaluation et démo GUI).
        """
        self._ensure_networks(env.state_size, env.action_size)
        actions = env.available_actions()

        state_t = torch.FloatTensor(env.get_state()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._actor(state_t).squeeze(0)

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
        log_dir: str = "runs/reinforce_critic",
        log_every: int = 500,
        save_callback=None,
    ) -> dict:
        """
        Entraîne acteur et critique, logue dans TensorBoard, évalue aux checkpoints.

        TensorBoard :
            train/mean_score, train/mean_length
            train/mean_actor_loss  : perte de l'acteur (−Σ log π · A_t)
            train/mean_critic_loss : perte MSE du critique (Σ (G_t − V(s_t))²)
            eval/mean_score, eval/mean_length, eval/mean_action_time_ms

        Returns:
            dict : métriques aux checkpoints
        """
        self._ensure_networks(env.state_size, env.action_size)

        writer             = SummaryWriter(log_dir=log_dir)
        checkpoint_results = {}
        checkpoints_done   = set()
        episode_scores     = []
        episode_lengths    = []
        actor_losses       = []
        critic_losses      = []
        best_score         = -float("inf")

        for episode in range(1, n_episodes + 1):
            log_probs, values, rewards = self._collect_episode(env)

            ep_score = sum(rewards)
            ep_steps = len(rewards)
            episode_scores.append(ep_score)
            episode_lengths.append(ep_steps)

            returns = self._compute_returns(rewards)
            a_loss, c_loss = self._update(log_probs, values, returns)
            actor_losses.append(a_loss)
            critic_losses.append(c_loss)

            if episode % log_every == 0:
                mean_score  = np.mean(episode_scores[-log_every:])
                mean_length = np.mean(episode_lengths[-log_every:])
                mean_aloss  = np.mean(actor_losses[-log_every:])
                mean_closs  = np.mean(critic_losses[-log_every:])

                writer.add_scalar("train/mean_score",       mean_score,  episode)
                writer.add_scalar("train/mean_length",      mean_length, episode)
                writer.add_scalar("train/mean_actor_loss",  mean_aloss,  episode)
                writer.add_scalar("train/mean_critic_loss", mean_closs,  episode)

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
                      f"actor_loss={np.mean(actor_losses[-500:]):.4f} | "
                      f"critic_loss={np.mean(critic_losses[-500:]):.4f}"
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
        """
        Déroule un épisode complet en suivant π_θ.
        Retourne log-probs (acteur), valeurs (critique) et récompenses.
        Les tenseurs log_probs et values conservent le graphe de calcul
        pour permettre la rétropropagation lors de la mise à jour.
        """
        state     = env.reset()
        log_probs = []
        values    = []
        rewards   = []

        while not env.is_game_over():
            actions = env.available_actions()
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # Acteur : politique
            logits = self._actor(state_t).squeeze(0)
            mask   = torch.full((env.action_size,), float("-inf"), device=self.device)
            for a in actions:
                mask[a] = 0.0
            log_probs_all = F.log_softmax(logits + mask, dim=-1)
            action_t = Categorical(log_probs_all.exp()).sample()
            log_probs.append(log_probs_all[action_t])

            # Critique : valeur de l'état courant
            value = self._critic(state_t).squeeze(0)
            values.append(value)

            state, reward, _ = env.step(action_t.item())
            rewards.append(reward)

        return log_probs, values, rewards

    def _compute_returns(self, rewards: list) -> list:
        """Retours Monte-Carlo discountés G_t = Σ_{k≥t} γ^(k-t) · r_k."""
        G       = 0.0
        returns = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns

    def _update(self, log_probs: list, values: list, returns: list):
        """
        Mise à jour conjointe de l'acteur et du critique.

        Critique → MSE : minimiser  Σ_t (G_t − V(s_t ; w))²
        Acteur   → PG  : maximiser  Σ_t log π(a_t | s_t ; θ) · (G_t − V(s_t ; w).détaché())

        Le détachement des valeurs dans la perte acteur est crucial :
        on veut que l'acteur profite de l'estimation du critique
        sans que le gradient de l'acteur ne perturbe l'apprentissage du critique.
        """
        returns_t = torch.FloatTensor(returns).to(self.device)
        values_t  = torch.stack(values)

        # Perte du critique (MSE sur les retours Monte-Carlo)
        critic_loss = F.mse_loss(values_t, returns_t)

        # Avantage vu par l'acteur — valeurs détachées du graphe du critique
        advantages = returns_t - values_t.detach()
        actor_loss = torch.stack([-lp * adv for lp, adv in zip(log_probs, advantages)]).sum()

        total_loss = actor_loss + self.value_coef * critic_loss

        self._optimizer.zero_grad()
        total_loss.backward()
        all_params = list(self._actor.parameters()) + list(self._critic.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        self._optimizer.step()

        return actor_loss.item(), critic_loss.item()

    # ── Sérialisation ─────────────────────────────────────────────────────────

    def get_state_dict(self) -> dict:
        return {
            "actor":        self._actor.state_dict()  if self._actor  else None,
            "critic":       self._critic.state_dict() if self._critic else None,
            "lr":           self.lr,
            "gamma":        self.gamma,
            "value_coef":   self.value_coef,
            "hidden_sizes": self.hidden_sizes,
        }

    def load_state_dict(self, data: dict, state_size: int, action_size: int) -> None:
        self.lr           = data["lr"]
        self.gamma        = data["gamma"]
        self.value_coef   = data.get("value_coef", 0.5)
        self.hidden_sizes = tuple(data["hidden_sizes"])
        self._ensure_networks(state_size, action_size)
        if data["actor"] is not None:
            self._actor.load_state_dict(data["actor"])
        if data["critic"] is not None:
            self._critic.load_state_dict(data["critic"])
