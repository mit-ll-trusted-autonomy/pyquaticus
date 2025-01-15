from utils import draw_result
from pyquaticus.base_policies.rrt.rrt_star import rrt_star
import matplotlib.pyplot as plt
import numpy as np
import time
start = np.array((10, 10))

triangles = []
for i in range(100):
    center = np.random.uniform(0, 1000, (2))
    triangles.append(np.array(((center - (15, 0)), (center + (0, 15)), (center + (15, 0)))))

squares = []
for i in range(100):
    center = np.random.uniform(0, 1000, (2))
    squares.append(np.array(((center - (15, 0)), (center + (0, 15)), (center + (15, 0)), (center - (0, 15)))))

start_time = time.time()
tree = rrt_star(start, np.array((900, 900)), triangles + squares, np.array(((0, 0), (1000, 1000))), agent_radius=2, max_step_size=20, num_iters=1000)
print(time.time() - start_time)
fig, ax = draw_result(tree, triangles + squares)
plt.show()

