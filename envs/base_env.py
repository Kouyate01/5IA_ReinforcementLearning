from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any
import numpy as np


class BaseEnv(ABC):
    """
    Classe de base abstraite pour tous les environnements du projet.
    
    Tout environnement DOIT implémenter ces méthodes pour être compatible
    avec n'importe quel agent (Random, DQL, MCTS, AlphaZero, etc.)
    """

    # -------------------------------------------------------------------------
    # Méthodes obligatoires (abstract)
    # -------------------------------------------------------------------------

    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Réinitialise l'environnement au début d'une nouvelle partie.
        
        Returns:
            np.ndarray: L'état initial encodé sous forme de vecteur numpy.
        """
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Effectue une action dans l'environnement.

        Args:
            action (int): L'index de l'action à jouer.

        Returns:
            Tuple:
                - np.ndarray : nouvel état encodé
                - float      : récompense obtenue
                - bool       : True si la partie est terminée
        """
        pass

    @abstractmethod
    def available_actions(self) -> List[int]:
        """
        Retourne la liste des actions légales dans l'état courant.

        Returns:
            List[int]: Liste des indices d'actions jouables.
        """
        pass

    @abstractmethod
    def is_game_over(self) -> bool:
        """
        Indique si la partie est terminée.

        Returns:
            bool: True si la partie est finie, False sinon.
        """
        pass

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """
        Retourne l'état courant encodé sous forme de vecteur numpy.
        C'est ce vecteur qui sera donné en entrée aux réseaux de neurones.

        Returns:
            np.ndarray: Vecteur d'état (flat, normalisé si possible).
        """
        pass

    @abstractmethod
    def clone(self) -> "BaseEnv":
        """
        Retourne une copie profonde de l'environnement courant.
        Indispensable pour MCTS et AlphaZero (simulation sans modifier l'état réel).

        Returns:
            BaseEnv: Copie indépendante de l'environnement.
        """
        pass

    @abstractmethod
    def render(self) -> None:
        """
        Affiche l'état courant dans le terminal (mode texte).
        Utilisé pour le debug et les logs.
        """
        pass

    # -------------------------------------------------------------------------
    # Propriétés obligatoires
    # -------------------------------------------------------------------------

    @property
    @abstractmethod
    def state_size(self) -> int:
        """
        Taille du vecteur d'état (dimension de l'entrée du réseau de neurones).
        
        Returns:
            int: Nombre de valeurs dans le vecteur d'état.
        """
        pass

    @property
    @abstractmethod
    def action_size(self) -> int:
        """
        Nombre total d'actions possibles dans l'environnement (pas forcément
        toutes légales à chaque instant).
        
        Returns:
            int: Taille de l'espace d'actions.
        """
        pass

    @property
    @abstractmethod
    def current_player(self) -> int:
        """
        Retourne l'index du joueur courant (0 ou 1 pour les jeux à 2 joueurs).

        Returns:
            int: Joueur courant.
        """
        pass

    # -------------------------------------------------------------------------
    # Méthodes optionnelles avec implémentation par défaut
    # -------------------------------------------------------------------------

    def score(self) -> float:
        """
        Score final de la partie pour le joueur 0 (ou le joueur principal).
        Utilisé pour les métriques d'évaluation.
        
        Par défaut retourne la dernière récompense — à surcharger si besoin.

        Returns:
            float: Score final (ex: 1.0 victoire, 0.5 nul, 0.0 défaite).
        """
        raise NotImplementedError("score() doit être implémenté pour les métriques d'évaluation")

    def get_action_mask(self) -> np.ndarray:
        """
        Retourne un masque booléen de taille action_size.
        True = action légale, False = action illégale.
        
        Très utile pour masquer les actions invalides dans les réseaux de neurones.

        Returns:
            np.ndarray: Tableau booléen de taille action_size.
        """
        mask = np.zeros(self.action_size, dtype=bool)
        for a in self.available_actions():
            mask[a] = True
        return mask

    def num_players(self) -> int:
        """
        Nombre de joueurs dans l'environnement.
        Par défaut 2 (jeux à 2 joueurs).

        Returns:
            int: Nombre de joueurs.
        """
        return 2

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"state_size={self.state_size}, "
            f"action_size={self.action_size})"
        )