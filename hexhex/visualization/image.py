import math

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw


def draw_board_image(board_tensor, file):
    red = (255,0,0)
    blue = (0,0,255)
    black = (0,0,0)
    white = (255,255,255)

    image = Image.new("RGB", (600, 370), white)

    draw = ImageDraw.Draw(image)

    r = 35

    for x in range(board_tensor.shape[1]):
        for y in range(board_tensor.shape[1]):
            angles = [math.pi / 6 + x * math.pi / 3 for x in range(6)]
            points = [(r + x*r/2 + y*r + math.cos(angle)*r/math.sqrt(3),
                    r + math.sqrt(3)/2*x*r + math.sin(angle)*r/math.sqrt(3))
                    for angle in angles]
            if board_tensor[0][x][y] == 1:
                draw.polygon(points, fill=red, outline=black)
            elif board_tensor[1][x][y] == 1:
                draw.polygon(points, fill=blue, outline=black)
            else:
                draw.polygon(points, fill=white, outline=black)
    image.save(file)
