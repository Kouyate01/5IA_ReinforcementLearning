class LineWorld:
    def __init__(self, size=5, start_state=2):
        """
        Environnement 1D : l'agent se déplace sur une ligne de `size` états.
        - Terminal aux extrémités (états 0 et size-1).
        - Actions : 0 = gauche, 1 = droite.
        """
        self.size = size
        self.start_state = start_state
        self.state = start_state
        self.terminal_states = [0, size - 1]
        self.action_space = [0, 1]

    def state_to_index(self, state):
        """Conversion état -> index (identique pour LineWorld)."""
        return state

    def index_to_state(self, index):
        """Conversion index -> état (identique pour LineWorld)."""
        return index

    @property
    def n_states(self):
        """Nombre total d'états."""
        return self.size

    def reset(self):
        """Reset l'environnement à l'état initial."""
        self.state = self.start_state
        return self.state

    def is_terminal(self, state):
        """Retourne True si l'état est terminal."""
        return state in self.terminal_states

    def get_reward(self, next_state):
        """Retourne la récompense pour l'état donné."""
        if next_state == 0:
            return -1.0
        elif next_state == self.size - 1:
            return 1.0
        else:
            return 0.0

    def step(self, action):
        """Applique l'action, avance l'agent et retourne (next_state, reward, done)."""
        if self.is_terminal(self.state):
            return self.state, 0.0, True

        if action == 0:
            next_state = max(self.state - 1, 0)
        elif action == 1:
            next_state = min(self.state + 1, self.size - 1)
        else:
            raise ValueError("Invalid action (0=left, 1=right)")

        reward = self.get_reward(next_state)
        done = self.is_terminal(next_state)
        self.state = next_state
        return next_state, reward, done

    def simulate_step(self, state, action):
        """Simule une action à partir d'un état, sans modifier self.state."""
        if self.is_terminal(state):
            return state, 0.0, True

        if action == 0:
            next_state = max(state - 1, 0)
        elif action == 1:
            next_state = min(state + 1, self.size - 1)
        else:
            raise ValueError("Invalid action (0=left, 1=right)")

        reward = self.get_reward(next_state)
        done = self.is_terminal(next_state)
        return next_state, reward, done

    def render(self):
        """Affiche la ligne avec l'agent (A) et les terminaux (T)."""
        line = []
        for i in range(self.size):
            if i == self.state:
                line.append('A')
            elif i in self.terminal_states:
                line.append('T')
            else:
                line.append('_')
        print(' '.join(line))

    def get_valid_actions(self, state):
        """
        Retourne la liste des actions valides depuis `state`.
        (0=left, 1=right)
        """
        if self.is_terminal(state):
            return []
        actions = []
        if state > 0:
            actions.append(0)  # gauche
        if state < self.size - 1:
            actions.append(1)  # droite
        return actions


if __name__ == "__main__":
    # demo interactive simple de LineWorld
    import numpy as np
    print("LineWorld demo:\n")
    env = LineWorld(size=7, start_state=3)
    env.reset()
    env.render()
    print("Actions valides en début :", env.get_valid_actions(env.state))
    # exécute une trajectoire aléatoire jusqu'à un état terminal
    while True:
        valid = env.get_valid_actions(env.state)
        if not valid:
            break
        action = np.random.choice(valid)
        action_name = "gauche" if action == 0 else "droite"
        next_state, reward, done = env.step(action)
        print(f"action {action_name} ({action}) -> état {next_state}, récompense {reward}, done={done}")
        env.render()
        if done:
            print("Atteint un état terminal.")
            break
