import pygame
import math

pygame.init()

screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Drawing a Part of a Circle")

white = (255, 255, 255)
red = (255, 0, 0)
blue = (0, 0, 255)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(white)

    # Define the bounding rectangle for the circle (a square)
    center_x, center_y = screen_width // 2, screen_height // 2
    radius = 100
    rect = (center_x - radius, center_y - radius, radius * 2, radius * 2)

    # Draw a quarter circle (from 0 to 90 degrees)
    # Angles are in radians, so 0 degrees is 0 radians, 90 degrees is math.pi / 2
    pygame.draw.arc(screen, red, rect, 0, math.pi / 2, 5)

    # Draw another quarter circle (from 180 to 270 degrees)
    pygame.draw.arc(screen, blue, rect, math.pi, 3 * math.pi / 2, 5)

    pygame.display.flip()

pygame.quit()