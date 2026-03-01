import pygame
import time

def visualize_lineworld_policy_interface(env, policy, cell_height=480):
    """
    Visualize policy for a LineWorld environment with UI (Next, Reset, Show Solution).
    - Case gauche = perdante (rouge), case droite = gagnante (verte).
    """
    pygame.init()
    size = 800
    n = env.size
    cell_w = size // n
    height = 800
    screen = pygame.display.set_mode((size, height))
    pygame.display.set_caption('LineWorld RL Visualizer')

    # Colors
    COLORS = {
        'agent': (255, 60, 60),
        'win': (60, 220, 60),      # Green for winning state (right)
        'lose': (220, 60, 60),     # Red for losing state (left)
        'empty': (230, 230, 230),
        'border': (100, 100, 100),
        'button_bg': (70, 130, 180),
        'button_fg': (240, 240, 240),
        'button_inactive': (170, 170, 170)
    }

    # Button setup
    font = pygame.font.SysFont(None, 36)
    BUTTONS = [
        {"label": "Next", "rect": pygame.Rect(30, height-60, 140, 48)},
        {"label": "Reset", "rect": pygame.Rect(190, height-60, 140, 48)},
        {"label": "Show Solution", "rect": pygame.Rect(350, height-60, 200, 48)},
        {"label": "Quit", "rect": pygame.Rect(570, height-60, 120, 48)}
    ]

    def draw_buttons(active=True):
        for btn in BUTTONS:
            color = COLORS['button_bg'] if active or btn['label'] == "Quit" else COLORS['button_inactive']
            pygame.draw.rect(screen, color, btn['rect'], border_radius=10)
            pygame.draw.rect(screen, COLORS['border'], btn['rect'], 2, border_radius=10)
            txt = font.render(btn['label'], True, COLORS['button_fg'])
            rect = txt.get_rect(center=btn['rect'].center)
            screen.blit(txt, rect)

    def draw_lineworld(state):
        screen.fill(COLORS['empty'])
        top = 40
        for i in range(n):
            rect = pygame.Rect(i * cell_w, top, cell_w, cell_height)
            if i == 0:
                color = COLORS['lose']    # Leftmost state: losing (red)
            elif i == n - 1:
                color = COLORS['win']     # Rightmost state: winning (green)
            else:
                color = COLORS['empty']
            pygame.draw.rect(screen, color, rect, border_radius=20)
            pygame.draw.rect(screen, COLORS['border'], rect, 2, border_radius=20)
        # Draw agent, big and centered vertically
        agent_y = top + cell_height // 2
        pygame.draw.circle(
            screen, COLORS['agent'],
            (state * cell_w + cell_w // 2, agent_y),
            min(cell_w, cell_height) // 3
        )

    # --- Main loop ---
    state = env.reset()
    done = False
    auto_solution = False
    action_trace = []
    running = True
    clock = pygame.time.Clock()

    while running:
        draw_lineworld(state)
        draw_buttons(active=not auto_solution)

        # Info display
        font2 = pygame.font.SysFont(None, 32)
        msg = f"Current state: {state} | Terminal: {done}"
        screen.blit(font2.render(msg, True, (30, 30, 30)), (20, 10))

        pygame.display.flip()

        if auto_solution and not done:
            time.sleep(0.25)
            action = policy[state]
            if action == -1:
                done = True
            else:
                state, reward, done = env.step(action)
                action_trace.append(state)
            continue
        elif auto_solution and done:
            auto_solution = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not auto_solution:
                for btn in BUTTONS:
                    if btn['rect'].collidepoint(event.pos):
                        if btn['label'] == "Next" and not done:
                            action = policy[state]
                            if action == -1:
                                done = True
                            else:
                                state, reward, done = env.step(action)
                                action_trace.append(state)
                        elif btn['label'] == "Reset":
                            state = env.reset()
                            done = False
                            action_trace = []
                        elif btn['label'] == "Show Solution" and not done:
                            auto_solution = True
                        elif btn['label'] == "Quit":
                            running = False


        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    # Demo : visualize a simple LineWorld policy
    from envs.line_world.line_world import LineWorld
    
    print("Initializing LineWorld visualization...")
    env = LineWorld(size=7, start_state=3)
    
    # Créer une politique simple : toujours aller à droite (1)
    # pour atteindre l'objectif gagnant (gain de +1)
    simple_policy = []
    for state in range(env.size):
        if env.is_terminal(state):
            simple_policy.append(-1)  # State terminal
        else:
            simple_policy.append(1)  # Always go right
    
    print(f"Policy: {simple_policy}")
    print("Launching visualization interface...")
    visualize_lineworld_policy_interface(env, simple_policy)