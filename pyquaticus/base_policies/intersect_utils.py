from re import L
import numpy as np
from typing import Union

LINE_INTERSECT_TOL = 1e-9


def point_in_polygon(point: np.ndarray, seglist: Union[np.ndarray, None]) -> bool:
    """
    Determines if a point is in a polygon

    Args:
        point (np.ndarray): (x, y)
        poly (np.ndarray): ((x1, y1), (x2, y2), ... , (xn, yn))

    Returns:
        bool: True if the point is inside the polygon
    """

    # TODO: Misses points on edges

    point = point.reshape((2))

    if seglist is None:
        return False
    seglist = seglist.reshape((-1, 2, 2))

    x1, y1 = (
        point[0],
        point[1],
    )
    x2, y2, x3, y3 = (
        seglist[:, 0, 0],
        seglist[:, 0, 1],
        seglist[:, 1, 0],
        seglist[:, 1, 1],
    )

    on_vertex = (y1 == y2) & (x1 == x2)

    if np.any(on_vertex):
        return True

    denom = y3 - y2

    intersect_x = ((y1 - y2) * (x3 - x2) / denom) + x2

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
    intersect = (((y1 < y2) != (y1 < y3)) & (x1 <= intersect_x))

    return bool(np.count_nonzero(intersect) % 2)


def point_in_polygons(point: np.ndarray, polys: np.ndarray) -> bool:
    point = point.reshape((2))
    polys = polys.reshape((polys.shape[0], -1, 2))
    for poly in polys:
        if point_in_polygon(point, poly):
            return True
    return False


def intersect(seg: np.ndarray, seglist: Union[np.ndarray, None]):
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
    # compute ray intersections
    x1, y1, x2, y2 = (
        seg[0, 0],
        seg[0, 1],
        seg[1, 0],
        seg[1, 1],
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
        & (intersect_x >= np.minimum(x1, x2) - LINE_INTERSECT_TOL)
        & (intersect_x <= np.maximum(x1, x2) + LINE_INTERSECT_TOL)
        & (intersect_y >= np.minimum(y1, y2) - LINE_INTERSECT_TOL)
        & (intersect_y <= np.maximum(y1, y2) + LINE_INTERSECT_TOL)
        & (intersect_x >= np.minimum(x3, x4) - LINE_INTERSECT_TOL)
        & (intersect_x <= np.maximum(x3, x4) + LINE_INTERSECT_TOL)
        & (intersect_y >= np.minimum(y3, y4) - LINE_INTERSECT_TOL)
        & (intersect_y <= np.maximum(y3, y4) + LINE_INTERSECT_TOL)
    )

    return np.any(intersect)


if __name__ == "__main__":

    # point = np.array((0, 1))
    # # Square
    # poly = np.array(((0, 0), (0, 1), (1, 1), (1, 0)))
    # # Diamond
    # # poly = np.array(((-1, 0), (0, 1), (1, 0), (0, -1)))
    # seglist = []
    # for i in range(len(poly)):
    #     seglist.append(poly[(i-1, i), :])
    # seglist = np.array(seglist)
    # print(point_in_polygon(point, seglist))

    import time
    import matplotlib.pyplot as plt

    num_sides = []
    poly_times = []
    int_times = []

    for i in range(3, 30000, 1000):
        point = np.random.uniform(0, 100, (1, 2))
        poly = np.random.uniform(0, 100, (i, 2))
        seglist = []
        for i in range(len(poly)):
            seglist.append(poly[(i-1, i), :])
        seglist = np.array(seglist)

        start_time = time.time()
        point_in_polygon(point, seglist)
        end_time = time.time()
        num_sides.append(i)
        poly_times.append(end_time - start_time)

        seg = np.random.uniform(0, 100, (2, 2))
        segs = np.random.uniform(0, 100, (i, 2, 2))

        start_time = time.time()
        intersect(seg, segs)
        end_time = time.time()
        int_times.append(end_time - start_time)
    fig, ax = plt.subplots()
    ax.scatter(num_sides, poly_times, c="r")
    ax.scatter(num_sides, int_times, c="b")
    plt.show()
