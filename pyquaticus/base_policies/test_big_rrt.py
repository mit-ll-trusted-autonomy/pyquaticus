from utils import rrt_star, draw_result
import numpy as np
start = np.array((10, 10))

triangles = []
for i in range(100):
    center = np.random.uniform(0, 1000, (2))
    triangles.append(np.array(((center - (15, 0)), (center + (0, 15)), (center + (15, 0)))))

squares = []
for i in range(100):
    center = np.random.uniform(0, 1000, (2))
    squares.append(np.array(((center - (15, 0)), (center + (0, 15)), (center + (15, 0)), (center - (0, 15)))))


tree = rrt_star(start, None, triangles + squares, np.array(((0, 0), (1000, 1000))), agent_radius=2, max_step_size=20)

draw_result(tree, triangles + squares)

