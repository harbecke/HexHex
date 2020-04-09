import math

import pygame

# Define the colors we will use in RGB format
from hex.utils.logger import logger

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK = (40, 40, 40)
LIGHT = (220, 220, 220)

STRAWBERRY = (251, 41, 67)
AZURE = (6, 154, 243)
BURGUNDY = (97, 0, 35)
ROYAL_BLUE = (5, 4, 170)


def _get_colors(dark_mode: bool):
    return {
        'DARK_MODE': dark_mode,
        'BACKGROUND': DARK if dark_mode else WHITE,
        'LINES': LIGHT if dark_mode else BLACK,
        'PLAYER_1': BURGUNDY if dark_mode else STRAWBERRY,
        'PLAYER_2': ROYAL_BLUE if dark_mode else AZURE
    }


class Gui:
    def __init__(self, board, radius, dark_mode=False):
        self.r = radius
        self.size = [int(self.r * (3 / 2 * board.size + 1)), int(self.r * (3 ** (1 / 2) / 2 * board.size + 1))]
        self.editor_mode = False  # AI will not move in editor mode
        self.colors = _get_colors(dark_mode)

        pygame.init()

        # Set the height and width of the screen
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("HexHex")

        self.clock = pygame.time.Clock()
        # distance of neighboring hexagons
        self.board = board
        self.update_board(board)
        pygame.font.init()
        self.font = pygame.font.SysFont(pygame.font.get_default_font(), int(radius / 2))

    def toggle_colors(self):
        self.colors = _get_colors(not self.colors['DARK_MODE'])

    def quit(self):
        # Be IDLE friendly
        pygame.quit()
        exit(0)

    def pixel_to_pos(self, pixel):
        positions = [(x, y) for x in range(self.board.size) for y in range(self.board.size)]

        def squared_distance(pos1, pos2):
            return (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2

        centers = [(position, self.get_center(position)) for position in positions]
        return min(centers, key=lambda x: squared_distance(x[1], pixel))[0]

    def get_center(self, pos):
        x = pos[0]
        y = pos[1]
        return [self.r + x * self.r / 2 + y * self.r, self.r + math.sqrt(3) / 2 * x * self.r]

    def update_board(self, board, field_text=None):
        # Clear the screen and set the screen background
        self.screen.fill(self.colors['BACKGROUND'])

        for x in range(board.size):
            for y in range(board.size):
                center = self.get_center([x, y])
                angles = [math.pi / 6 + x * math.pi / 3 for x in range(6)]
                points = [[center[0] + math.cos(angle) * self.r / math.sqrt(3),
                           center[1] + math.sin(angle) * self.r / math.sqrt(3)]
                          for angle in angles]

                if board.get_owner((x, y)) == 0:
                    pygame.draw.polygon(self.screen, self.colors['PLAYER_1'], points, 0)
                elif board.get_owner((x, y)) == 1:
                    pygame.draw.polygon(self.screen, self.colors['PLAYER_2'], points, 0)
                pygame.draw.polygon(self.screen, self.colors['LINES'], points, 3)

                if field_text is not None:
                    field_text_pos = board.player * (x * board.size + y) + \
                                     (1 - board.player) * (y * board.size + x)
                    text = field_text[field_text_pos]
                    textsurface = self.font.render(f'{text}', True, self.colors['LINES'])
                    text_size = self.font.size(text)
                    self.screen.blit(textsurface, (center[0] - text_size[0] // 2,
                                                   center[1] - text_size[1] // 2))

        # Go ahead and update the screen with what we've drawn.
        # This MUST happen after all the other drawing commands.
        pygame.display.flip()

    def wait_for_click(self):
        while True:
            self.clock.tick(10)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # If user clicked close
                    self.quit()
                    exit(0)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    return

    def get_move(self):
        while True:
            # This limits the while loop to a max of 10 times per second.
            # Leave this out and we will use all CPU we can.
            self.clock.tick(10)

            for event in pygame.event.get():  # User did something
                if event.type == pygame.QUIT:  # If user clicked close
                    self.quit()
                    exit(0)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    return self.pixel_to_pos(event.pos)
                if event.type == pygame.KEYDOWN and event.unicode == 'd':
                    self.toggle_colors()
                    return 'redraw'
                if event.type == pygame.KEYDOWN and event.unicode == 'a':
                    return 'ai_move'
                if event.type == pygame.KEYDOWN and event.unicode == 'z':
                    return 'undo_move'
                if event.type == pygame.KEYDOWN and event.unicode == 'e':
                    self.editor_mode = not self.editor_mode
                    logger.info(f'Editor mode: {self.editor_mode}')
