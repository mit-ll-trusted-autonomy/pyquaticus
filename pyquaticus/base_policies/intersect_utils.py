from re import L
import numpy as np
from typing import Union


def point_in_polygon(point: np.ndarray, seglist: Union[np.ndarray, None], radius: float = 1e-9) -> bool:
    """
    Determines if a point is inside (or on the edges/vertices) of a polygon

    Args:
        point (np.ndarray): [x, y]
        seglist (np.ndarray): array of shape (n, 2, 2) for an n-sided polygon
        where seglist[i] = [[x1, y1], [x2, y2]] for the i-th edge of the polygon

    Returns:
        bool: True if the point is inside the polygon
    """

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

    # Only need to check one of the points in each segment
    # because every point will show up once in each position
    on_vertex = (np.linalg.norm(y1 - y2) <= radius) & (np.linalg.norm(x1 - x2) <= radius)

    if np.any(on_vertex):
        return True

    denom = y3 - y2
    intersect_x = ((y1 - y2) * (x3 - x2) / denom) + x2

    on_edge = (np.any(np.abs(x1 - intersect_x) <= radius) & ((y1 < y2) != (y1 < y3))) | ((denom == 0) & (np.abs(y1 - y2) <= radius) & ((x1 < x2) != (x1 < x3)))

    if np.any(on_edge):
        print("on edge")
        print(np.abs(x1 - intersect_x))
        return True

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
    intersect = (((y1 < y2) != (y1 < y3)) & (x1 <= intersect_x + radius))

    print(intersect)

    return bool(np.count_nonzero(intersect) % 2)


def point_in_polygons(point: np.ndarray, polys: np.ndarray, radius: float = 1e-9) -> bool:
    point = point.reshape((2))
    polys = polys.reshape((polys.shape[0], -1, 2))
    for poly in polys:
        seglist = []
        for i in range(len(poly)):
            seglist.append(poly[(i-1, i), :])
        seglist = np.array(seglist)
        if point_in_polygon(point, seglist, radius):
            print(f"{point} in {poly}")
            return True
    return False


def intersect(seg: np.ndarray, seglist: Union[np.ndarray, None], radius: float = 1e-9):
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
    # compute ray intersections
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


if __name__ == "__main__":

    # point = np.array((0, 0))
    # segs = np.array((((0, -40), (0, 0)), ((4, 4), (5, 5))))
    # # Square
    # # poly = np.array(((0, 0), (0, 1), (1, 1), (1, 0)))
    # shape = np.array(((20, 15), (50, -5), (45, -15), (25, -5), (20, -15), (25, -25), (20, -35), (10, -15)))
    # # Diamond
    # # poly = np.array(((-1, 0), (0, 1), (1, 0), (0, -1)))
    # seglist = []
    # for i in range(len(shape)):
    #     seglist.append(shape[(i-1, i), :])
    # seglist = np.array(seglist)
    # print(point_in_polygon(point, seglist))
    # print(intersect(segs[0], seglist))

    end = np.array((5, 10))
    obstacles = np.array(
        (((4, 4), (4, 7), (7, 7), (7, 4)))
    ).reshape(1, 4, 2)
    print(point_in_polygons(end, obstacles, 2))
