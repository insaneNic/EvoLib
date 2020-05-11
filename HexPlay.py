import pygame, sys
import pygame.freetype
from backend.Hexplode import *
from backend.agent import *
from backend.funcs import softmax


# Defining function to draw hexagons
def draw_hexagon(surface, color, mid_point, radius = 100):
    points = [mid_point] * 6
    for i in range(6):
        points[i] = [mid_point[0] + radius * np.sin(np.pi * i / 3),
                     mid_point[1] + radius * np.cos(np.pi * i / 3)]

    pygame.draw.polygon(surface, color, points)


# Ratio of short diagonal to long diagonal
RATIO = np.sqrt(3) / 2

# Initializing Board and window parameters
BOARD_SIZE = 5
MAGIC = 2 * BOARD_SIZE - 1
HEX_DIAMETER = 100
WIDTH = MAGIC * HEX_DIAMETER
HEIGHT = int(MAGIC * HEX_DIAMETER * RATIO)
SIZE = (WIDTH, HEIGHT)

# Creating list of hexagon positions
pos_list = [[]] * MAGIC

for i in range(0, BOARD_SIZE):
    pos_list[i] = [
        [WIDTH // 2 - (BOARD_SIZE + i - 1) * HEX_DIAMETER * RATIO // 2 + HEX_DIAMETER * RATIO * j,
         RATIO * RATIO * (HEX_DIAMETER // 2 + HEX_DIAMETER * i) + HEX_DIAMETER // 2]
        for j in range(BOARD_SIZE + i)]
    pos_list[MAGIC - (i + 1)] = [
        [WIDTH // 2 - (BOARD_SIZE + i - 1) * HEX_DIAMETER * RATIO // 2 + HEX_DIAMETER * RATIO * j,
         RATIO * RATIO * (HEX_DIAMETER * (MAGIC - i - 0.5)) + HEX_DIAMETER // 2]
        for j in range(BOARD_SIZE + i)]

# Flatten the list
pos_list = [np.array(pos) for row in pos_list for pos in row]

# Loading agent
agent = Agent([], softmax)
agent.load_weights('new_best')

# Initializing Game
game = HexIterGame(BOARD_SIZE)

# Initializing pyGame
pygame.init()

screen = pygame.display.set_mode(SIZE)
pygame.display.set_caption("Hexplode!")

pygame.font.init()
font = pygame.freetype.Font('backend/ahronbd.ttf', 16)

pygame.display.flip()

clicked = [0 for _ in pos_list]

# Starting Game loop
j = 0
while 1:

    # If previous move was human
    if j % 2 == 1:
        game.agent_move(agent)
        j += 1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            # Get Click location and check distance to closest hex
            click_loc = np.array(pygame.mouse.get_pos())
            distances = list(map(lambda x, y: np.linalg.norm(x-y),
                                 list(zip([click_loc] * len(pos_list))), pos_list))

            # If close enough allow click
            if np.min(distances) <= RATIO * HEX_DIAMETER / 2:
                clicked_hex = np.argmin(distances)
                if not game.human_move(clicked_hex):
                    j += 1
            print("You Clicked at: " + str(click_loc))

    # Start rendering
    screen.fill((10, 10, 10))

    cur_board = game.get_lin_board()
    for p, pos, clc in zip(cur_board, pos_list, clicked):
        if p != 0:
            draw_hexagon(screen, (150 + 60 * (p < 0), 150, 150 + 60 * (p > 0)), pos, radius = HEX_DIAMETER // 2 - 2)
            font.render_to(screen, pos, str(p), (240, 240, 240), size = 36)
        else:
            draw_hexagon(screen, (150, 150, 150 + 60 * clc), pos, radius = HEX_DIAMETER // 2 - 2)

    font.render_to(screen, [10, 10], "Turn: " + str(j), (200, 200, 200), size = 20)

    pygame.display.update()
