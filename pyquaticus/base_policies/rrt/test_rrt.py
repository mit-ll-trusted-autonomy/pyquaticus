import numpy as np
from pyquaticus.base_policies.rrt.rrt_star import rrt_star
from utils import draw_result
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

start = np.array((0, 0))

obstacles = [np.array(
    (((4., 4), (4, 7), (7, 7), (7, 4)))
)]

area = np.array(((-2, -2), (11, 11)))

tree = rrt_star(start, np.array((10, 10)), obstacles, area, 2, 5, 1000)

fig, ax = draw_result(tree, obstacles)
ax.plot((2, 2, 9, 9, 2), (2, 9, 9, 2, 2), "r")
ax.add_patch(Circle((4, 4), 2, fill=False, edgecolor="r"))
ax.add_patch(Circle((4, 7), 2, fill=False, edgecolor="r"))
ax.add_patch(Circle((7, 4), 2, fill=False, edgecolor="r"))
ax.add_patch(Circle((7, 7), 2, fill=False, edgecolor="r"))
plt.show()