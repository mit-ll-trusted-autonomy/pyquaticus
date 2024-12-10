from attr import dataclass
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from typing import SupportsFloat


@dataclass
class Point:
    pos: np.ndarray
    cost: float = 0.0
    parent: Union["Point", None] = None


def point_in_polygon(point: np.ndarray, poly: np.ndarray) -> bool:
    """
    Determines if a point is in a polygon (or on the border)

    Args:
        point (np.ndarray): (x, y)
        poly (np.ndarray): ((x1, y1), (x2, y2), ... , (xn, yn))

    Returns:
        bool: True if the point is inside or on the border of the polygon
    """

    point = point.reshape((2))
    poly = poly.reshape((-1, 2))

    # Check if test point lies on a vertex
    if point.tolist() in poly.tolist():
        return True

    inside = False

    # For each segment in the polygon
    for i in range(poly.shape[0]):

        seg = poly[(i - 1, i), :]

        # Check if test point's y value lies between segment's y values
        # (or equal to the segment's lowest y value)
        #
        #
        # So that this only counts as one intersection:
        #
        #                  0---
        #                 /
        #                /
        #   p ----------0------->
        #                \
        #                 \
        #                  0---
        #

        if (point[1] < seg[0][1]) != (point[1] < seg[1][1]):

            # Find where segment intersects horizontal line through test point
            x = (
                (point[1] - seg[0][1])
                * (seg[1][0] - seg[0][0])
                / (seg[1][1] - seg[0][1])
            ) + seg[0][0]

            # If test point is exactly on the intersection point,
            # then it is on the boundary of the polygon
            if point[0] == x:
                return True
            # If test point is to the left of the intersection point,
            # then we have an intersection
            if point[0] <= x:
                inside = not inside
    return inside


def point_in_polygons(point: np.ndarray, polys: np.ndarray) -> bool:
    point = point.reshape((2))
    polys = polys.reshape((polys.shape[0], -1, 2))
    for poly in polys:
        if point_in_polygon(point, poly):
            return True
    return False


def rrt_star(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: Union[np.ndarray, None],
    area: np.ndarray,
    num_iters: int = 100,
) -> Union[list[Point], None]:

    points = [Point(start, 0, None)]
    goal_point = Point(goal, 0, None)
    for i in range(num_iters):
        new_point = get_random_point(area, obstacles)
        nearest = get_nearest(new_point, points)
        new_point.parent = nearest
        new_point.cost = nearest.cost + dist(new_point, nearest)
        near_points = get_near(new_point, points, 10)
        choose_parent(new_point, near_points)
        rewire(new_point, near_points)
        points.append(new_point)

    return points


def get_near(point: Point, points: list[Point], radius: float) -> list[Point]:
    near_points = []
    for p in points:
        if dist(point, p) <= radius:
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
        print(f"point {point} parent {point.parent}")
        if point.parent is not None:
            ax.plot([point.pos[0], point.parent.pos[0]], [point.pos[1], point.parent.pos[1]], "b")
        else:
            print(f"point {point} has no parent")
    # for point in points:
    #     ax.plot(point.pos[0], point.pos[1], "ko")
    if obstacles is not None:
        for obstacle in obstacles:
            for i in range(obstacle.shape[0]):
                seg = obstacle[(i - 1, i), :]
                ax.plot(seg[:, 0], seg[:, 1], "r")
    plt.show()


def get_nearest(point: Point, points: list[Point]):
    return min(points, key=lambda p: dist(p, point))


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

    # import time
    # import matplotlib.pyplot as plt

    # num_sides = []
    # times = []

    # for i in range(3, 30000, 1000):
    #     point = np.random.uniform(0, 100, (1, 2))
    #     poly = np.random.uniform(0, 100, (i, 2))

    #     start_time = time.time()
    #     point_in_polygon(point, poly)
    #     end_time = time.time()
    #     num_sides.append(i)
    #     times.append(end_time - start_time)

    # plt.scatter(num_sides, times)
    # plt.show()

    # point = np.array((-1, 0))
    # poly = np.array(((-1, -1), (-1, 1), (1, 1), (1, -1)))
    # print(point_in_polygon(point, poly))
    start = np.array((0, 0))
    end = np.array((10, 10))
    obstacles = np.array((((5, 5), (5, 6), (6, 6), (6, 5)), ((1, 1), (1, 2), (2, 2), (2, 1))))
    area = np.array(((-1, -1), (11, 11)))
    tree = rrt_star(start, end, obstacles, area)
    if tree is not None:
        draw_result(tree, area, obstacles)
