from attr import dataclass
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt


np.seterr(divide="ignore")


@dataclass
class Point:
    pos: np.ndarray
    cost: float = 0.0
    parent: Optional["Point"] = None


def get_near(
    point: Point,
    points: list[Point],
    radius: float,
    seglist: Optional[np.ndarray],
    agent_radius: float,
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
    points: list[Point],
    obstacles: Optional[list[np.ndarray]],
):
    fig, ax = plt.subplots()
    for point in points:
        if point.parent is not None:
            ax.plot(
                [point.pos[0], point.parent.pos[0]],
                [point.pos[1], point.parent.pos[1]],
                "b",
            )

    if obstacles is not None:
        for obstacle in obstacles:
            for i in range(obstacle.shape[0]):
                seg = obstacle[(i - 1, i), :]
                ax.plot(seg[:, 0], seg[:, 1], "r")

    return fig, ax


def get_nearest(
    point: Point,
    points: list[Point],
    seglist: Optional[np.ndarray],
    agent_radius: float,
) -> Optional[Point]:
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
    grouped_seglist: Optional[np.ndarray],
    ungrouped_seglist: Optional[np.ndarray],
    points: list[Point],
    max_step_size: float,
    agent_radius: float,
    goal: Optional[np.ndarray],
) -> tuple[Point, Point]:
    """
    Gets a random point (and its nearest neighbor) that is not in any of the obstacles.

    Args:
        area (np.ndarray): ((xmin, ymin), (xmax, ymax))
        obstacles (np.ndarray): list of obstacles

    Returns:
        Point: point not in obstacles
    """

    # 5% chance to sample goal point, if it exists
    if (np.random.uniform(0, 1) < 0.05) and goal is not None:
        rand_point = Point(goal)
    else:
        rand_point = Point(np.random.uniform(area[0], area[1], (2)))


    if grouped_seglist is None:
        nearest = get_nearest(rand_point, points, ungrouped_seglist, agent_radius)
        assert nearest is not None
        new_point = bound(rand_point, nearest, max_step_size)
        return new_point, nearest

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
    return np.pad(
        seglist,
        ((0, length - seglist.shape[0]), (0, 0), (0, 0)),
        constant_values=np.nan,
    )


def point_in_polygons(
    point: np.ndarray, seglists: Optional[np.ndarray], radius: float = 1e-9
):
    """
    Determines if a point is in any of the polygons, provided as an array of segments

    Note: pad all polygons with np.nan so that the length of each individual seglist is the same.
    The function get_grouped_seglist() does this.

    point.shape should be (2)

    seglists.shape should be (n, k, 2, 2), where there are n polygons with a maximum of k edges

    Args:
        point (np.ndarray): (x, y)

        seglists (np.ndarray):
                               ((((x1, y1), (x2, y2)), ((x2, y2), (x3, y3)), ... , ((xk, yk), (x1, y1))                 -> first polygon

                                (((x1, y1), (x2, y2)), ((x2, y2), (x3, y3)), ... , ((np.nan, np.nan), (np.nan, np.nan)) -> second polygon

                                ...

                                ...

                                (((x1, y1), (x2, y2)), ((x2, y2), (x3, y3)), ... , ((np.nan, np.nan), (np.nan, np.nan))) -> nth polygon

    Returns:
        bool: True if the point is inside any of the polygons (or within the given radius of an edge or vertex)
    """
    point = point.reshape((2))

    if seglists is None:
        return False

    x1, y1 = (
        point[0],
        point[1],
    )
    x2, y2, x3, y3 = (
        seglists[:, :, 0, 0],
        seglists[:, :, 0, 1],
        seglists[:, :, 1, 0],
        seglists[:, :, 1, 1],
    )

    vertex_dists = np.linalg.norm(
        np.array((((x1 - x2).flatten()), ((y1 - y2).flatten()))), axis=0
    )
    if np.any(vertex_dists <= radius):
        return True

    denom = y3 - y2
    intersect_x = ((y1 - y2) * (x3 - x2) / denom) + x2

    on_edge = ((np.abs(x1 - intersect_x) <= radius) & ((y1 < y2) != (y1 < y3))) | (
        (denom == 0) & (np.abs(y1 - y2) <= radius) & ((x1 < x2) != (x1 < x3))
    )
    if np.any(on_edge):
        return True

    intersect = ((y1 < y2) != (y1 < y3)) & (x1 <= intersect_x + radius)

    return np.any(np.count_nonzero(intersect, axis=1) % 2)


def intersect(seg: np.ndarray, seglist: Optional[np.ndarray], radius: float = 1e-9):
    """
    Determines if a segment intersects any of the segments in an array of segments

    Note: if segments are parallel, this function will not detect an intersection

    Args:
        seg (np.ndarray): ((x1, y1), (x2, y2))
        seglist (np.ndarray): (((x1, y1), (x2, y2)), ((x1, y1), (x2, y2)), ... )

    Returns:
        bool: True if the segment intersects any of the segments in the array
    """
    if seglist is None:
        return False

    seglist = seglist.reshape((-1, 2, 2))
    seg = seg.reshape((-1, 2, 2))

    x1, y1, x2, y2 = (
        seg[:, 0, 0].reshape((-1, 1)),
        seg[:, 0, 1].reshape((-1, 1)),
        seg[:, 1, 0].reshape((-1, 1)),
        seg[:, 1, 1].reshape((-1, 1)),
    )
    x3, y3, x4, y4 = (
        seglist[:, 0, 0],
        seglist[:, 0, 1],
        seglist[:, 1, 0],
        seglist[:, 1, 1],
    )

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    intersect_x = (
        (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
    ) / denom
    intersect_y = (
        (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
    ) / denom

    intersect = (
        (denom != 0)
        & (intersect_x >= np.minimum(x1, x2) - radius)
        & (intersect_x <= np.maximum(x1, x2) + radius)
        & (intersect_y >= np.minimum(y1, y2) - radius)
        & (intersect_y <= np.maximum(y1, y2) + radius)
        & (intersect_x >= np.minimum(x3, x4) - radius)
        & (intersect_x <= np.maximum(x3, x4) + radius)
        & (intersect_y >= np.minimum(y3, y4) - radius)
        & (intersect_y <= np.maximum(y3, y4) + radius)
    )

    return np.any(intersect, axis=1)

def intersect_new(seg: np.ndarray, seglist: Optional[np.ndarray], radius: float = 1e-9):
    """
    Determines if a segment intersects any of the segments in an array of segments

    Note: if segments are parallel, this function will not detect an intersection

    Args:
        seg (np.ndarray): ((x1, y1), (x2, y2))
        seglist (np.ndarray): (((x1, y1), (x2, y2)), ((x1, y1), (x2, y2)), ... )

    Returns:
        bool: True if the segment intersects any of the segments in the array
    """
    if seglist is None:
        return np.full((seg.shape[0]), False)
    
    pts = seglist[:, 0, :]

    past1 = (np.matmul(pts - seg[:, 1, :].reshape((-1, 1, 2)), (seg[:, 1, :] - seg[:, 0, :]).reshape((-1, 2, 1))) > 0).reshape((seg.shape[0], pts.shape[0]))
    past0 = (np.matmul(pts - seg[:, 0, :].reshape((-1, 1, 2)), (seg[:, 1, :] - seg[:, 0, :]).reshape((-1, 2, 1))) < 0).reshape((seg.shape[0], pts.shape[0]))
    between = np.logical_and(np.logical_not(past1), np.logical_not(past0))

    num = np.abs(np.cross((pts - seg[:, 0, :].reshape((-1, 1, 2))).transpose((0, 2, 1)).ravel(order="F").reshape((-1, 2, seg.shape[0])).transpose((0, 2, 1)), seg[:, 1, :] - seg[:, 0, :]))
    denom = np.linalg.norm(seg[:, 1, :] - seg[:, 0, :], axis=1)

    dist0 = np.linalg.norm(pts - seg[:, 0, :].reshape((-1, 1, 2)), axis=2)
    dist1 = np.linalg.norm(pts - seg[:, 1, :].reshape((-1, 1, 2)), axis=2)
    distline = (num / denom).transpose((1, 0))

    distfinal = past1 * dist1 + past0 * dist0 + between * distline

    seglist = seglist.reshape((-1, 2, 2))
    seg = seg.reshape((-1, 2, 2))

    x1, y1, x2, y2 = (
        seg[:, 0, 0].reshape((-1, 1)),
        seg[:, 0, 1].reshape((-1, 1)),
        seg[:, 1, 0].reshape((-1, 1)),
        seg[:, 1, 1].reshape((-1, 1)),
    )
    x3, y3, x4, y4 = (
        seglist[:, 0, 0],
        seglist[:, 0, 1],
        seglist[:, 1, 0],
        seglist[:, 1, 1],
    )

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    intersect_x = (
        (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
    ) / denom
    intersect_y = (
        (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
    ) / denom

    intersect = (
        (denom != 0)
        & (intersect_x >= np.minimum(x1, x2) - radius)
        & (intersect_x <= np.maximum(x1, x2) + radius)
        & (intersect_y >= np.minimum(y1, y2) - radius)
        & (intersect_y <= np.maximum(y1, y2) + radius)
        & (intersect_x >= np.minimum(x3, x4) - radius)
        & (intersect_x <= np.maximum(x3, x4) + radius)
        & (intersect_y >= np.minimum(y3, y4) - radius)
        & (intersect_y <= np.maximum(y3, y4) + radius)
    )

    return np.logical_or(np.any(intersect, axis=1), np.any(distfinal < radius, axis=1))
