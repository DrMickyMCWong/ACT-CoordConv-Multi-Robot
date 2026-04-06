# Create checkerboard PNG with Python:
from PIL import Image, ImageDraw
import numpy as np

# Create 800x800 checkerboard (8x8 grid, 100px per square)
size = 800
square_size = 100
img = Image.new('RGB', (size, size), 'white')
draw = ImageDraw.Draw(img)

for i in range(8):
    for j in range(8):
        if (i + j) % 2 == 1:  # Black squares
            x0 = i * square_size
            y0 = j * square_size
            draw.rectangle([x0, y0, x0+square_size, y0+square_size], fill='black')

img.save('/home/hk/Documents/isaac_sim/checkerboard.png')