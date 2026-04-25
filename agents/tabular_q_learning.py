import numpy as np
import random
import time
from torch.utils.tensorboard import SummaryWriter
import torch
from envs.base_env import BaseEnv


class TabularQLearning:
    """
    Agent Tabular Q-Learning avec TensorBoard, checkpoints et meilleur modèle.

    Applicable sur :
        - LineWorld  (7 états)
        - GridWorld  (25 états)
        - TicTacToe  (à vérifier)
        x Bobail : espace d'états trop grand

    ALGORITHME
    A chaque step :
        1. Choisir une action avec politique epsilon-greedy
        2. Executer l'action -> obtenir (s', r, done)
        3. Mise a jour de Bellman :
            Q(s,a) <- Q(s,a) + alpha x [r + gamma x max_a' Q(s',a') - Q(s,a)]
            td_error = r + gamma x max Q(s',a') - Q(s,a)   <- loss loggée
        4. Decrementer epsilon

    HYPERPARAMETRES
        alpha         : taux d'apprentissage          defaut 0.1
        gamma         : facteur de discount            defaut 0.99
        epsilon       : exploration initiale           defaut 1.0
        epsilon_min   : exploration minimale           defaut 0.01
        epsilon_decay : decroissance par episode       defaut 0.995
    """

    CHECKPOINTS = [1_000, 10_000, 100_000, 1_000_000]

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self._q_table: dict = {}

    @classmethod
    def from_config(cls, config: dict) -> "TabularQLearning":
        """
        Instancie l'agent depuis un dictionnaire de configuration.
        Les clés manquantes prennent leur valeur par défaut.

        Exemple :
            agent = TabularQLearning.from_config({
                "alpha": 0.05,
                "gamma": 0.95,
                "epsilon_decay": 0.999,
            })
        """
        return cls(
            alpha         = config.get("alpha",         0.1),
            gamma         = config.get("gamma",         0.99),
            epsilon       = config.get("epsilon",       1.0),
            epsilon_min   = config.get("epsilon_min",   0.01),
            epsilon_decay = config.get("epsilon_decay", 0.995),
        )

    # ─────────────────────────────────────────────────────────────
    # Interface principale
    # ─────────────────────────────────────────────────────────────

    def select_action(self, env: BaseEnv, greedy: bool = False) -> int:
        """
        Politique epsilon-greedy.
        greedy=True -> exploitation pure (evaluation et demo GUI).
        """
        actions = env.available_actions()
        if not greedy and random.random() < self.epsilon:
            return random.choice(actions)
        state_key = self._state_to_key(env.get_state())
        q_values  = self._get_q_values(state_key, env.action_size)
        return max(actions, key=lambda a: q_values[a])

    def train(
        self,
        env: BaseEnv,
        n_episodes: int = 100_000,
        eval_episodes: int = 500,
        log_dir: str = "runs/tabular_q_learning",
        log_every: int = 500,
        save_callback=None,   # appelé à chaque checkpoint : save_callback(agent, checkpoint)
    ) -> dict:
        """
        Entraine l'agent, logue dans TensorBoard et evalue aux checkpoints.

        TensorBoard :
            train/mean_score    : score moyen glissant pendant l'entrainement
            train/mean_length   : longueur moyenne des parties
            train/epsilon       : valeur courante de epsilon
            train/q_table_size  : nombre d'etats dans la Q-table
            train/mean_td_error : erreur TD moyenne (loss) <- courbe de loss
            eval/mean_score     : score greedy au checkpoint
            eval/mean_length    : longueur greedy au checkpoint
            eval/mean_action_time_ms
            eval/q_table_size

        Args:
            save_callback : fonction optionnelle appelée à chaque checkpoint
                            pour sauvegarder le modèle depuis train.py.
                            Signature : save_callback(agent, checkpoint, is_best)

        Returns:
            dict : metriques aux checkpoints
        """
        writer             = SummaryWriter(log_dir=log_dir)
        checkpoint_results = {}
        checkpoints_done   = set()
        episode_scores     = []
        episode_lengths    = []
        td_errors          = []     # liste des erreurs TD pour la loss
        best_score         = -float("inf")

        for episode in range(1, n_episodes + 1):
            # ── Episode d'entrainement ────────────────────────────
            state     = env.reset()
            state_key = self._state_to_key(state)
            ep_score  = 0.0
            ep_steps  = 0

            while not env.is_game_over():
                action = self.select_action(env)
                next_state, reward, done = env.step(action)
                next_key = self._state_to_key(next_state)

                td_error = self._update(
                    state_key    = state_key,
                    action       = action,
                    reward       = reward,
                    next_key     = next_key,
                    done         = done,
                    action_size  = env.action_size,
                    next_actions = env.available_actions(),
                )
                td_errors.append(abs(td_error))

                state_key = next_key
                ep_score += reward
                ep_steps += 1

            self._decay_epsilon()
            episode_scores.append(ep_score)
            episode_lengths.append(ep_steps)

            # ── Log TensorBoard (courbes d'entrainement) ──────────
            if episode % log_every == 0:
                mean_score    = np.mean(episode_scores[-log_every:])
                mean_length   = np.mean(episode_lengths[-log_every:])
                mean_td_error = np.mean(td_errors[-log_every * 10:]) if td_errors else 0.0

                writer.add_scalar("train/mean_score",    mean_score,         episode)
                writer.add_scalar("train/mean_length",   mean_length,        episode)
                writer.add_scalar("train/epsilon",       self.epsilon,       episode)
                writer.add_scalar("train/q_table_size",  len(self._q_table), episode)
                writer.add_scalar("train/mean_td_error", mean_td_error,      episode)

            # ── Evaluation aux checkpoints ────────────────────────
            if episode in self.CHECKPOINTS and episode not in checkpoints_done:
                metrics = self.evaluate(env, n_episodes=eval_episodes)
                checkpoint_results[episode] = metrics
                checkpoints_done.add(episode)

                writer.add_scalar("eval/mean_score",          metrics["mean_score"],         episode)
                writer.add_scalar("eval/mean_length",         metrics["mean_length"],        episode)
                writer.add_scalar("eval/mean_action_time_ms", metrics["mean_action_time_ms"],episode)
                writer.add_scalar("eval/q_table_size",        metrics["q_table_size"],       episode)

                # Sauvegarde checkpoint + meilleur modèle
                is_best = metrics["mean_score"] > best_score
                if is_best:
                    best_score = metrics["mean_score"]

                if save_callback is not None:
                    save_callback(self, episode, is_best)

                print(f"[Checkpoint {episode:>8}] "
                      f"score={metrics['mean_score']:.3f} | "
                      f"length={metrics['mean_length']:.1f} | "
                      f"td_error={np.mean(td_errors[-500:]):.4f} | "
                      f"Q-table={metrics['q_table_size']} etats"
                      f"{' ← BEST' if is_best else ''}")

        writer.close()
        return checkpoint_results

    def evaluate(self, env: BaseEnv, n_episodes: int = 500) -> dict:
        """
        Evalue la policy en mode GREEDY (epsilon=0).
        Produit les metriques demandees par le sujet.
        """
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
            "q_table_size":        len(self._q_table),
        }

    def reset_exploration(self, epsilon: float = 1.0) -> None:
        self.epsilon = epsilon

    # ─────────────────────────────────────────────────────────────
    # Méthodes privées
    # ─────────────────────────────────────────────────────────────

    def _state_to_key(self, state: np.ndarray):
        if state.sum() == 1.0 and state.max() == 1.0:
            return int(np.argmax(state))
        return tuple(state.tolist())

    def _get_q_values(self, state_key, action_size: int) -> np.ndarray:
        if state_key not in self._q_table:
            self._q_table[state_key] = np.zeros(action_size, dtype=np.float32)
        return self._q_table[state_key]

    def _update(self, state_key, action, reward, next_key, done,
                action_size, next_actions) -> float:
        """
        Mise a jour de Bellman.
        Retourne l'erreur TD (utilisee comme loss dans TensorBoard).
        """
        q_current = self._get_q_values(state_key, action_size)

        if done or len(next_actions) == 0:
            target = reward
        else:
            q_next    = self._get_q_values(next_key, action_size)
            best_next = max(q_next[a] for a in next_actions)
            target    = reward + self.gamma * best_next

        td_error = target - q_current[action]
        q_current[action] += self.alpha * td_error
        return td_error

    def _decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)