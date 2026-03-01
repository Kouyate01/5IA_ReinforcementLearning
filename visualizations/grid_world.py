import pygame
import time

def visualize_gridworld_policy_interface(env, policy, cell_size=120, margin=40, btn_bar_height=120):
    """
    Visualize policy for a GridWorld environment with a pro interface.
    - Big window, pro buttons, centered grid, no overlap.
    """
    pygame.init()
    nrows, ncols = env.n_rows, env.n_cols

    # --- Compute true display size (center grid) ---
    grid_width = ncols * cell_size
    grid_height = nrows * cell_size
    width = max(1100, grid_width + 2 * margin)
    height = grid_height + btn_bar_height + 2 * margin

    grid_x = (width - grid_width) // 2
    grid_y = margin

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('GridWorld RL Visualizer')
    clock = pygame.time.Clock()

    # Colors
    COLORS = {
        'bg': (245, 245, 245),
        'cell': (240, 240, 240),
        'border': (80, 80, 80),
        'terminal': (60, 210, 80),
        'agent': (60, 120, 255),
        'text': (30, 30, 30),
        'button_bg': (70, 130, 180),
        'button_fg': (240, 240, 240),
        'button_inactive': (170, 170, 170),
        'arrow': (60, 60, 60)
    }

    # Arrow directions
    ARROWS = {
        0: ((0.5,0.75),(0.5,0.25)), # up
        1: ((0.5,0.25),(0.5,0.75)), # down
        2: ((0.75,0.5),(0.25,0.5)), # left
        3: ((0.25,0.5),(0.75,0.5)), # right
    }
    ARROW_HEADS = {
        0: [(0.46,0.3),(0.54,0.3),(0.5,0.15)],
        1: [(0.46,0.7),(0.54,0.7),(0.5,0.85)],
        2: [(0.3,0.46),(0.3,0.54),(0.15,0.5)],
        3: [(0.7,0.46),(0.7,0.54),(0.85,0.5)],
    }
    font = pygame.font.SysFont(None, 32)
    font_big = pygame.font.SysFont(None, 40)

    # Buttons
    btn_bar_top = grid_y + grid_height + 20
    btns = [
        {"label": "Next", "rect": pygame.Rect(width//2-320, btn_bar_top, 120, 56)},
        {"label": "Reset", "rect": pygame.Rect(width//2-180, btn_bar_top, 120, 56)},
        {"label": "Show Solution", "rect": pygame.Rect(width//2-20, btn_bar_top, 200, 56)},
        {"label": "Quit", "rect": pygame.Rect(width//2+200, btn_bar_top, 120, 56)},
    ]

    def draw_buttons(active=True):
        for btn in btns:
            color = COLORS['button_bg'] if active or btn['label'] == "Quit" else COLORS['button_inactive']
            pygame.draw.rect(screen, color, btn['rect'], border_radius=12)
            pygame.draw.rect(screen, COLORS['border'], btn['rect'], 2, border_radius=12)
            txt = font.render(btn['label'], True, COLORS['button_fg'])
            rect = txt.get_rect(center=btn['rect'].center)
            screen.blit(txt, rect)

    def draw_grid(state):
        screen.fill(COLORS['bg'])
        # Grid cells
        for row in range(nrows):
            for col in range(ncols):
                x = grid_x + col * cell_size
                y = grid_y + row * cell_size
                rect = pygame.Rect(x, y, cell_size, cell_size)
                if (row, col) in env.terminal_states:
                    color = COLORS['terminal']
                else:
                    color = COLORS['cell']
                pygame.draw.rect(screen, color, rect, border_radius=14)
                pygame.draw.rect(screen, COLORS['border'], rect, 3)
                # Arrow (policy)
                idx = env.state_to_index((row, col))
                a = policy[idx]
                if (row, col) not in env.terminal_states and a in ARROWS:
                    start, end = ARROWS[a]
                    sx, sy = x + start[0]*cell_size, y + start[1]*cell_size
                    ex, ey = x + end[0]*cell_size, y + end[1]*cell_size
                    pygame.draw.line(screen, COLORS['arrow'], (sx, sy), (ex, ey), 7)
                    points = [(x + px*cell_size, y + py*cell_size) for px, py in ARROW_HEADS[a]]
                    pygame.draw.polygon(screen, COLORS['arrow'], points)
        # Agent
        row, col = state
        ax = grid_x + col * cell_size + cell_size//2
        ay = grid_y + row * cell_size + cell_size//2
        pygame.draw.circle(screen, COLORS['agent'], (ax, ay), int(cell_size*0.28))
        # Text info
        info_y = btn_bar_top - 48
        txt = font_big.render(f"Current state: {state}", True, COLORS['text'])
        screen.blit(txt, (grid_x, info_y))
        # You can add reward display or path display here

    # --- Main loop ---
    state = env.reset()
    done = False
    auto_solution = False
    running = True
    clock = pygame.time.Clock()

    while running:
        draw_grid(state)
        draw_buttons(active=not auto_solution)
        pygame.display.flip()

        if auto_solution and not done:
            time.sleep(0.22)
            idx = env.state_to_index(state)
            action = policy[idx]
            if action == -1 or env.is_terminal(state):
                done = True
            else:
                state, reward, done = env.step(action)
            continue
        elif auto_solution and done:
            auto_solution = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not auto_solution:
                for btn in btns:
                    if btn['rect'].collidepoint(event.pos):
                        if btn['label'] == "Next" and not done:
                            idx = env.state_to_index(state)
                            action = policy[idx]
                            if action == -1 or env.is_terminal(state):
                                done = True
                            else:
                                state, reward, done = env.step(action)
                        elif btn['label'] == "Reset":
                            state = env.reset()
                            done = False
                        elif btn['label'] == "Show Solution" and not done:
                            auto_solution = True
                        elif btn['label'] == "Quit":
                            running = False
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    # Demo : visualize a simple GridWorld policy
    from envs.grid_world.grid_world import GridWorld
    
    print("Initializing GridWorld visualization...")
    env = GridWorld(n_rows=4, n_cols=4, start_state=(0, 0))
    
    # Créer une politique simple : aller à droite (3) puis en bas (1)
    # Pour atteindre l'objectif en bas-à-droite (3,3)
    simple_policy = []
    for idx in range(env.n_states):
        state = env.index_to_state(idx)
        row, col = state
        
        if env.is_terminal(state):
            simple_policy.append(-1)  # State terminal
        else:
            # Priorité : bas (1) si pas à la dernière ligne, sinon droite (3)
            if row < env.n_rows - 1:
                simple_policy.append(1)  # down
            elif col < env.n_cols - 1:
                simple_policy.append(3)  # right
            else:
                simple_policy.append(-1)
    
    print(f"Policy: {simple_policy}")
    print("Launching visualization interface...")
    visualize_gridworld_policy_interface(env, simple_policy)