from pyquaticus.base_policies.rrt.utils import *
import numpy as np

seg = np.array((((0, 5), (5, 5)))).reshape((1, 2, 2))
seg = np.array((((0, 5), (5, 5)), ((0, 0), (7, 0))))
seg = np.array((((0, 5), (5, 5)), ((0, 0), (7, 0)), ((0, -1), (7, -1)), ((0.9, 4), (2, 5.1))))

poly = np.array(((2, 2), (2, 4), (6, 4), (6, 2)))

seglist = get_ungrouped_seglist([poly])

print(intersect(seg, seglist, 1))

print(intersect_new(seg, seglist, 1))