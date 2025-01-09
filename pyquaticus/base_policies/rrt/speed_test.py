import time
import matplotlib.pyplot as plt
import numpy as np
from pyquaticus.base_policies.rrt.utils import point_in_polygons, intersect, get_grouped_seglist, get_ungrouped_seglist, intersect_new

num_sides = []
poly_times = []
int_times = []
int_new_times = []
count = 0
for i in range(3, 300000, 10000):
    point = np.random.uniform(0, 100, (1, 2))
    poly = np.random.uniform(0, 100, (i, 2))
    grouped_seglist = get_grouped_seglist([poly])
    ungrouped_seglist = get_ungrouped_seglist([poly])

    start_time = time.time()
    point_in_polygons(point, grouped_seglist)
    end_time = time.time()
    num_sides.append(i)
    poly_times.append(end_time - start_time)

    seg = np.random.uniform(0, 100, (2, 2)).reshape(1, 2, 2)

    start_time = time.time()
    old = intersect(seg, ungrouped_seglist)
    end_time = time.time()
    int_times.append(end_time - start_time)

    start_time = time.time()
    new = intersect_new(seg, ungrouped_seglist)
    end_time = time.time()
    int_new_times.append(end_time - start_time)

    count += old != new

print(count)
fig, ax = plt.subplots()
ax.scatter(num_sides, poly_times, c="r")
ax.scatter(num_sides, int_times, c="b")
ax.scatter(num_sides, int_new_times, c="g")
plt.show()