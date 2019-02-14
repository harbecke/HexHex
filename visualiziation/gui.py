import pygame
import math
 
def draw_board(board):
    # Initialize the game engine
    pygame.init()
     
    # Define the colors we will use in RGB format
    BLACK = (  0,   0,   0)
    WHITE = (255, 255, 255)
    BLUE =  (  0,   0, 255)
    GREEN = (  0, 255,   0)
    RED =   (255,   0,   0)
    PLAYER_1 = RED#(123, 52, 123)
    PLAYER_2 = BLUE#(255, 255, 128)

    # Set the height and width of the screen
    size = [600, 370]
    screen = pygame.display.set_mode(size)
     
    pygame.display.set_caption("HexHex")
     
    #Loop until the user clicks the close button.
    done = False
    clock = pygame.time.Clock()

    while not done:
        # This limits the while loop to a max of 10 times per second.
        # Leave this out and we will use all CPU we can.
        clock.tick(10)
         
        for event in pygame.event.get(): # User did something
            if event.type == pygame.QUIT: # If user clicked close
                done=True # Flag that we are done so we exit this loop
     
        # All drawing code happens after the for loop and but
        # inside the main while done==False loop.
         
        # Clear the screen and set the screen background
        screen.fill(WHITE)
        
        r = 35

        for x in range(board.size):
            for y in range(board.size):
                angles = [math.pi / 6 + x * math.pi / 3 for x in range(6)]
                points = [[r + x*r/2 + y*r +math.cos(angle)*r/math.sqrt(3),
                        r + math.sqrt(3)/2*x*r + math.sin(angle)*r/math.sqrt(3)] 
                        for angle in angles]
                if board.board_tensor[0][x][y] == 1:
                    pygame.draw.polygon(screen, PLAYER_1, points,0)
                elif board.board_tensor[1][x][y] == 1:
                    pygame.draw.polygon(screen, PLAYER_2, points,0)
                pygame.draw.polygon(screen, BLACK, points,3)          


             
        # Go ahead and update the screen with what we've drawn.
        # This MUST happen after all the other drawing commands.
        pygame.display.flip()
    # Be IDLE friendly
    pygame.quit()