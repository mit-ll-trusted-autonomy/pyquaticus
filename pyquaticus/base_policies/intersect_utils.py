import numpy as np
from typing import Optional

def point_in_polygons(point: np.ndarray, seglists: Optional[np.ndarray], radius: float = 1e-9):
    """
    Determines if a point is in any of the polygons, provided as an array of segments

    Note: pad all polygons with np.nan so that the length of each individual seglist is the same

    point.shape should be (2)
    seglists.shape should be (n, k, 2, 2), where there are n polygons with a maximum of k edges

    Args:
        point (np.ndarray): (x, y)
        seglists (np.ndarray): ((((x1, y1), (x2, y2)), ((x2, y2), (x3, y3)), ... , ((xk, yk), (x1, y1))                 -> first polygon
                                (((x1, y1), (x2, y2)), ((x2, y2), (x3, y3)), ... , ((np.nan, np.nan), (np.nan, np.nan)) -> second polygon
                                ...
                                ...
                                (((x1, y1), (x2, y2)), ((x2, y2), (x3, y3)), ... , ((np.nan, np.nan), (np.nan, np.nan))) -> nth polygon

    Returns:
        bool: True if the segment intersects any of the segments in the array
    """
    point = point.reshape((2))

    if seglists is None:
        return False
    # seglist = seglists.reshape((-1, 2, 2))

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

    x_dist = x1 - x2
    y_dist = y1 - y2
    vec = np.array(((x_dist.flatten()), (y_dist.flatten())))
    dists = np.linalg.norm(vec, axis=0)
    if np.any(dists <= radius):
        return True

    denom = y3 - y2
    intersect_x = ((y1 - y2) * (x3 - x2) / denom) + x2

    on_edge = ((np.abs(x1 - intersect_x) <= radius) & ((y1 < y2) != (y1 < y3))) | ((denom == 0) & (np.abs(y1 - y2) <= radius) & ((x1 < x2) != (x1 < x3)))
    if np.any(on_edge):
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
    def get_grouped_seglist(polys: list[np.ndarray]) -> np.ndarray:
        seglists = []
        max_len = 3
        for poly in polys:
            if poly.shape[0] > max_len:
                max_len = poly.shape[0]
            seglist = []
            for i in range(poly.shape[0]):
                seglist.append(poly[(i - 1, i), :])
            seglists.append(np.array(seglist))
        padded_seglists = []
        for seglist in seglists:
            padded_seglists.append(pad(seglist, max_len))
        return np.array(padded_seglists)
    def pad(seglist: np.ndarray, length: int) -> np.ndarray:
        return np.pad(seglist, ((0, length - seglist.shape[0]), (0, 0), (0, 0)), constant_values=np.nan)
    point = np.array((0, 0))
    # segs = np.array((((0, -40), (0, 0)), ((4, 4), (5, 5))))
    # Square
    square = np.array(((0., 0), (0, 1), (1, 1), (1, 0)))
    # shape = np.array(((20, 15), (50, -5), (45, -15), (25, -5), (20, -15), (25, -25), (20, -35), (10, -15)))
    # Diamond
    diamond = np.array(((-1., 0), (0, 1), (1, 0), (0, -1)))
    seglist = get_grouped_seglist([square, diamond])
    print(point_in_polygons(point, seglist))