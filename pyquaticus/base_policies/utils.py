from attr import dataclass
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from pyquaticus.base_policies.intersect_utils import point_in_polygons, intersect


@dataclass
class Point:
    pos: np.ndarray
    cost: float = 0.0
    parent: Union["Point", None] = None


def rrt_star(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: Union[np.ndarray, None],
    area: np.ndarray,
    num_iters: int = 1000,
) -> Union[list[Point], None]:

    points = [Point(start, 0, None)]
    seglist = None
    if obstacles is not None:
        seglist = []
        for obstacle in obstacles:
            for i in range(obstacle.shape[0]):
                seglist.append(obstacle[(i - 1, i), :])
        seglist = np.array(seglist)

    for i in range(num_iters):
        new_point = get_random_point(area, obstacles)
        nearest = get_nearest(new_point, points, seglist)
        if nearest is None:
            continue
        new_point.parent = nearest
        new_point.cost = nearest.cost + dist(new_point, nearest)
        near_points = get_near(new_point, points, 1, seglist)
        choose_parent(new_point, near_points)
        rewire(new_point, near_points)
        points.append(new_point)

    return points


def get_near(point: Point, points: list[Point], radius: float, seglist: Union[np.ndarray, None]) -> list[Point]:
    near_points = []
    for p in points:
        if (dist(point, p) <= radius) and not intersect(np.array([point.pos, p.pos]), seglist):
            near_points.append(p)
    return near_points


def choose_parent(point: Point, parents: list[Point]):
    for parent in parents:
        if parent.cost + dist(parent, point) < point.cost:
            point.cost = parent.cost + dist(parent, point)
            point.parent = parent


def rewire(potential_parent: Point, near_points: list[Point]):
    for point in near_points:
        if potential_parent.cost + dist(potential_parent, point) < point.cost:
            point.cost = potential_parent.cost + dist(potential_parent, point)
            point.parent = potential_parent


def draw_result(points: list[Point], area: np.ndarray, obstacles: Union[np.ndarray, None]):
    fig, ax = plt.subplots()
    for point in points:
        if point.parent is not None:
            ax.plot([point.pos[0], point.parent.pos[0]], [point.pos[1], point.parent.pos[1]], "b")
    # for point in points:
    #     ax.plot(point.pos[0], point.pos[1], "ko")
    if obstacles is not None:
        for obstacle in obstacles:
            for i in range(obstacle.shape[0]):
                seg = obstacle[(i - 1, i), :]
                ax.plot(seg[:, 0], seg[:, 1], "r")
    plt.show()


def get_nearest(point: Point, points: list[Point], seglist: Union[np.ndarray, None]) -> Union[Point, None]:
    if seglist is None:
        return min(points, key=lambda p: dist(p, point))
    min_dist = np.inf
    min_point = None
    for p in points:
        if not intersect(np.array([point.pos, p.pos]), seglist):
            if dist(p, point) < min_dist:
                min_dist = dist(p, point)
                min_point = p
    return min_point


def get_random_point(area: np.ndarray, obstacles: Union[np.ndarray, None]) -> Point:
    """
    Gets a random point that is not in any of the obstacles.

    Args:
        area (np.ndarray): ((xmin, ymin), (xmax, ymax))
        obstacles (np.ndarray): list of obstacles

    Returns:
        Point: point not in obstacles
    """
    if obstacles is None:
        return Point(np.random.uniform(area[0], area[1], (2)))
    point = np.random.uniform(area[0], area[1], (2))
    while point_in_polygons(point, obstacles):
        point = np.random.uniform(area[0], area[1], (2))
    return Point(point)


def dist(p1: Point, p2: Point) -> float:
    return float(np.linalg.norm(p1.pos - p2.pos))


if __name__ == "__main__":

    start = np.array((0, 0))
    end = np.array((10, 10))
    obstacles = np.array((((4, 4), (4, 7), (7, 7), (7, 4)), ((1, 1), (1, 5), (5, 5), (5, 1))))
    area = np.array(((-1, -1), (11, 11)))
    tree = rrt_star(start, end, obstacles, area)
    if tree is not None:
        draw_result(tree, area, obstacles)
