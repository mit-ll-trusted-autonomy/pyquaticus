import numpy as np

from pyquaticus.utils.utils import angle180


def vec_to_heading(vec):
    """Converts a vector to a magnitude and heading (deg)."""
    angle = np.degrees(np.arctan2(vec[1], vec[0]))
    return angle180(angle)


def bearing_to_vec(heading):
    return [np.cos(np.deg2rad(heading)), np.sin(np.deg2rad(heading))]


def rb_to_rect(point: np.ndarray) -> np.ndarray:
    """Returns the rectangular coordinates of polar point `point`."""
    dist = point[0]
    bearing = point[1]
    unit_vec = bearing_to_vec(bearing)
    return np.array([dist * unit_vec[0], dist * unit_vec[1]], dtype=np.float64)


def get_avoid_vect(avoid_pos, avoid_threshold=10.0):
    """
    This function finds the vector most pointing away to all enemy agents.

    Args:
        agent: An agents position
        avoid_pos: All other agent (polar) positions we potentially need to avoid.
        avoid_threshold: The threshold that, when the agent is closer than this range,
            is attempted to be avoided.

    Returns
    -------
        np.array vector that points away from as many agents as possible
    """
    avoid_vects = []
    need_avoid = False
    for avoid_ag in avoid_pos:
        if avoid_ag[0] < avoid_threshold:
            coeff = np.divide(avoid_threshold, avoid_ag[0])
            ag_vect = rel_bearing_to_local_unit_rect(avoid_ag[1])
            avoid_vects.append([coeff * ag_vect[0], coeff * ag_vect[1]])
            need_avoid = True
    av_x = 0.0
    av_y = 0.0
    if need_avoid:
        for vects in avoid_vects:
            av_x += vects[0]
            av_y += vects[1]
        norm = np.linalg.norm(np.array([av_x, av_y]))
        final_avoid_unit_vect = np.array(
            [-1.0 * np.divide(av_x, norm), -1.0 * np.divide(av_y, norm)]
        )
    else:
        final_avoid_unit_vect = np.array([0, 0])

    return final_avoid_unit_vect


def unit_vect_between_points(start: np.ndarray, end: np.ndarray):
    """Calculates the unit vector between two rectangular points."""
    return (end - start) / np.linalg.norm(end - start)


def global_rect_to_abs_bearing(vec):
    """Calculates the absolute bearing of a rectangular vector in the global frame."""
    return angle180(90 - np.degrees(np.arctan2(vec[1], vec[0])))


def dist_rel_bearing_to_local_rect(dist, rel_bearing):
    """Calculates the local frame rectangular vector given a distance and relative bearing."""
    return dist * rel_bearing_to_local_unit_rect(rel_bearing)


def rel_bearing_to_local_unit_rect(rel_bearing):
    """Calculates the local frame rectangular unit vector in the direction of the given relative bearing."""
    rad = np.deg2rad(rel_bearing)
    return np.array((np.sin(rad), np.cos(rad)))


def local_rect_to_rel_bearing(vec):
    """Calculates the relative bearing of a rectangular vector in the local frame."""
    return global_rect_to_abs_bearing(vec)
