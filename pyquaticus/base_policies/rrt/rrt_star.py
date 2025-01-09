import numpy as np
from typing import Optional
from pyquaticus.base_policies.rrt.utils import *

def rrt_star(
    start: np.ndarray,
    goal: Optional[np.ndarray],
    obstacles: Optional[list[np.ndarray]],
    area: np.ndarray,
    agent_radius: float = 1e-9,
    max_step_size: float = 2,
    num_iters: int = 1000,
) -> list[Point]:
    
    np.seterr(all="ignore")

    points = [Point(start, 0, None)]

    # Generate array of all obstacle segments
    ungrouped_seglist = None
    grouped_seglist = None
    if obstacles is not None:
        ungrouped_seglist = get_ungrouped_seglist(obstacles)
        grouped_seglist = get_grouped_seglist(obstacles)

    for _ in range(num_iters):

        new_point, nearest = get_random_point(
            area, grouped_seglist, ungrouped_seglist, points, max_step_size, agent_radius, goal
        )

        new_point.parent = nearest
        new_point.cost = nearest.cost + dist(new_point, nearest)

        near_points = get_near(new_point, points, max_step_size, ungrouped_seglist, agent_radius)

        choose_parent(new_point, near_points)

        rewire(new_point, near_points)

        points.append(new_point)

    return points

