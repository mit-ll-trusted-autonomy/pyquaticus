from attr import dataclass
import numpy as np
from typing import Union, Optional
import matplotlib.pyplot as plt
from pyquaticus.base_policies.intersect_utils import point_in_polygons, intersect
import time


@dataclass
class Point:
    pos: np.ndarray
    cost: float = 0.0
    parent: Union["Point", None] = None





def get_near(
    point: Point, points: list[Point], radius: float, seglist: Union[np.ndarray, None], agent_radius: float,
) -> list[Point]:
    seg_array = np.array([np.array([p.pos, point.pos]) for p in points])
    dist_array = np.linalg.norm(seg_array[:, 0, :] - seg_array[:, 1, :], axis=1)
    int_array = intersect(seg_array, seglist, agent_radius)

    near_array = (dist_array <= radius) & (np.logical_not(int_array))
    near_points = []
    for i in range(len(points)):
        if near_array[i]:
            near_points.append(points[i])

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


def draw_result(
    points: list[Point], obstacles: Optional[list[np.ndarray]],
):
    fig, ax = plt.subplots()
    for point in points:
        if point.parent is not None:
            ax.plot(
                [point.pos[0], point.parent.pos[0]],
                [point.pos[1], point.parent.pos[1]],
                "b",
            )
    # for point in points:
    #     ax.plot(point.pos[0], point.pos[1], "ko")
    if obstacles is not None:
        for obstacle in obstacles:
            for i in range(obstacle.shape[0]):
                seg = obstacle[(i - 1, i), :]
                ax.plot(seg[:, 0], seg[:, 1], "r")
    return fig, ax


def get_nearest(
    point: Point, points: list[Point], seglist: Union[np.ndarray, None], agent_radius: float,
) -> Union[Point, None]:
    if seglist is None:
        return min(points, key=lambda p: dist(p, point))
    seg_array = np.array([np.array([p.pos, point.pos]) for p in points])
    dist_array = np.linalg.norm(seg_array[:, 0, :] - seg_array[:, 1, :], axis=1)
    int_array = intersect(seg_array, seglist, agent_radius)

    if np.all(int_array):
        return None

    min_index = np.argmin(dist_array + np.max(dist_array) * int_array)

    return points[min_index]


def get_random_point(
    area: np.ndarray,
    grouped_seglist: Union[np.ndarray, None],
    ungrouped_seglist: Union[np.ndarray, None],
    points: list[Point],
    max_step_size: float,
    agent_radius: float,
) -> tuple[Point, Point]:
    """
    Gets a random point (and its nearest neighbor) that is not in any of the obstacles.

    Args:
        area (np.ndarray): ((xmin, ymin), (xmax, ymax))
        obstacles (np.ndarray): list of obstacles

    Returns:
        Point: point not in obstacles
    """
    if grouped_seglist is None:
        rand_point = Point(np.random.uniform(area[0], area[1], (2)))
        nearest = get_nearest(rand_point, points, ungrouped_seglist, agent_radius)
        assert nearest is not None
        new_point = bound(rand_point, nearest, max_step_size)
        return new_point, nearest

    rand_point = Point(np.random.uniform(area[0], area[1], (2)))
    nearest = get_nearest(rand_point, points, ungrouped_seglist, agent_radius)
    while nearest is None:
        rand_point = Point(np.random.uniform(area[0], area[1], (2)))
        nearest = get_nearest(rand_point, points, ungrouped_seglist, agent_radius)
    new_point = bound(rand_point, nearest, max_step_size)
    while point_in_polygons(new_point.pos, grouped_seglist, agent_radius):
        rand_point = Point(np.random.uniform(area[0], area[1], (2)))
        nearest = get_nearest(rand_point, points, ungrouped_seglist, agent_radius)
        while nearest is None:
            rand_point = Point(np.random.uniform(area[0], area[1], (2)))
            nearest = get_nearest(rand_point, points, ungrouped_seglist, agent_radius)
        new_point = bound(rand_point, nearest, max_step_size)
    return new_point, nearest


def bound(to_point: Point, from_point: Point, max_step_size):
    vector = to_point.pos - from_point.pos
    if np.linalg.norm(vector) > max_step_size:
        vector = vector * max_step_size / np.linalg.norm(vector)
        to_point.pos = from_point.pos + vector
    return to_point


def dist(p1: Point, p2: Point) -> float:
    return float(np.linalg.norm(p1.pos - p2.pos))

def get_seglist(poly: np.ndarray) -> np.ndarray:
    seglist = []
    for i in range(poly.shape[0]):
        seglist.append(poly[(i - 1, i), :])
    seglist = np.array(seglist)
    return seglist

def get_ungrouped_seglist(polys: list[np.ndarray]) -> np.ndarray:
    seglist = []
    for poly in polys:
        for i in range(poly.shape[0]):
            seglist.append(poly[(i - 1, i), :])
    seglist = np.array(seglist)
    return seglist

def get_grouped_seglist(polys: list[np.ndarray]) -> np.ndarray:
    seglists = []
    max_len = 3
    for poly in polys:
        if poly.shape[0] > max_len:
            max_len = poly.shape[0]
        seglist = []
        for i in range(poly.shape[0]):
            seglist.append(poly[(i - 1, i), :].astype(np.float64))
        seglists.append(np.array(seglist))
    padded_seglists = []
    for seglist in seglists:
        padded_seglists.append(pad(seglist, max_len))
    return np.array(padded_seglists)

def pad(seglist: np.ndarray, length: int) -> np.ndarray:
    return np.pad(seglist, ((0, length - seglist.shape[0]), (0, 0), (0, 0)), constant_values=np.nan)





