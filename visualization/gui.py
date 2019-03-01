import pygame
import math

# Define the colors we will use in RGB format
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
PLAYER_1 = RED  # (123, 52, 123)
PLAYER_2 = BLUE  # (255, 255, 128)

class Gui:
    def __init__(self, board, radius):
        self.r = radius
        self.size = [int(self.r*(3/2*board.size+1)), int(self.r*(3**(1/2)/2*board.size+1))]

        pygame.init()

        # Set the height and width of the screen
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("HexHex")

        self.clock = pygame.time.Clock()
         # distance of neighboring hexagons
        self.board = board
        self.update_board(board)
        pygame.font.init()
        self.font = pygame.font.SysFont(pygame.font.get_default_font(), 25)

    def quit(self):
        # Be IDLE friendly
        pygame.quit()
        exit(0)

    def pixel_to_pos(self, pixel):
        positions = [(x, y) for x in range(self.board.size) for y in range(self.board.size)]
        def squared_distance(pos1, pos2):
            return (pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2
        centers = [(position, self.get_center(position)) for position in positions]
        return min(centers, key=lambda x : squared_distance(x[1], pixel))[0]

    def get_center(self, pos):
        x = pos[0]
        y = pos[1]
        return [self.r + x * self.r / 2 + y * self.r, self.r + math.sqrt(3) / 2 * x * self.r]

    def update_board(self, board, field_text=None):
        self.board = board

        # Clear the screen and set the screen background
        self.screen.fill(WHITE)

        for x in range(board.size):
            for y in range(board.size):
                center = self.get_center([x,y])
                angles = [math.pi / 6 + x * math.pi / 3 for x in range(6)]
                points = [[center[0] + math.cos(angle)*self.r/math.sqrt(3),
                           center[1] + math.sin(angle)*self.r/math.sqrt(3)]
                        for angle in angles]
                if board.board_tensor[0][x][y] == 1:
                    pygame.draw.polygon(self.screen, PLAYER_1, points,0)
                elif board.board_tensor[1][x][y] == 1:
                    pygame.draw.polygon(self.screen, PLAYER_2, points,0)
                pygame.draw.polygon(self.screen, BLACK, points,3)

                if field_text is not None:
                    text = field_text[x * board.size + y]
                    textsurface = self.font.render(f'{text}', True, (0, 0, 0))
                    self.screen.blit(textsurface, (center[0]-self.r/6, center[1] - self.r/6))

        # Go ahead and update the screen with what we've drawn.
        # This MUST happen after all the other drawing commands.
        pygame.display.flip()

    def get_cell(self):
        while True:
            # This limits the while loop to a max of 10 times per second.
            # Leave this out and we will use all CPU we can.
            self.clock.tick(10)

            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    self.quit()
                    exit(0)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    return self.pixel_to_pos(event.pos)
