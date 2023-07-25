# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for
# Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the
# author(s) and do not necessarily reflect the views of the Under Secretary of Defense
# for Research and Engineering.
#
# (C) 2023 Massachusetts Institute of Technology.
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS
# Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S.
# Government rights in this work are defined by DFARS 252.227-7013 or DFARS
# 252.227-7014 as detailed above. Use of this work other than as specifically
# authorized by the U.S. Government may violate any copyrights that exist in this
# work.

# SPDX-License-Identifier: BSD-3-Clause

import math

import numpy as np
import pygame
from numpy.linalg import norm
from sympy import Circle, Line, Point, Ray, Segment, Triangle


def rot2d(vector: np.ndarray, theta) -> np.ndarray:
    """
    This function takes a 2D vector whose tail is at (0,0), and
    an angle theta (radians) as input. It moves the vector by theta
    degrees with respect to the origin (around the unit circle) and
    outputs the resulting vector endpoint in cartesian coordinates.
    """
    assert vector.shape == (2,), "rot2d requires a 2D vector as input"

    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    return np.dot(rot, vector)


def rc_intersection(
    ray: np.ndarray, circle_center: np.ndarray, circle_radius
) -> np.ndarray:
    """
    Calculates the point at which a ray intersects and exits a circle, assuming this point exists.
    Returns exit_wound.

    ray: np.array([ [source_x, source_y], [end_x, end_y] ])
    2D ray that propegates in the direction such that it passes through
    (end_x, end_y).

    circle_center: np.array([x, y])

    exit_wound: np.array([x, y])
    """
    ray_position_vector = ray[1] - ray[0]
    ray_magnitude = norm(ray_position_vector)

    if ray_magnitude == 0:
        ray = np.array([circle_center, ray[0]])
        return rc_intersection(ray, circle_center, circle_radius)

    ray_unit = ray_position_vector / ray_magnitude

    x1, y1, x2, y2 = ray.flatten()

    h = circle_center[0]
    k = circle_center[1]
    r = (
        circle_radius + 0.1
    )  # adding buffer to calculation of intersection with keepout zone (fixes rounding error bug)

    x_int = np.zeros((2, 1))
    y_int = np.zeros((2, 1))

    if x1 == x2:
        x_int[:] = x1
        y_int[0] = k + np.sqrt(r**2 - (x_int[0] - h) ** 2)
        y_int[1] = k - np.sqrt(r**2 - (x_int[1] - h) ** 2)
        intersections = np.concatenate((x_int, y_int), axis=1)

    elif y1 == y2:
        y_int[:] = y1
        x_int[0] = h + np.sqrt(r**2 - (y_int[0] - k) ** 2)
        x_int[1] = h - np.sqrt(r**2 - (y_int[1] - k) ** 2)
        intersections = np.concatenate((x_int, y_int), axis=1)

    else:
        m = (y2 - y1) / (x2 - x1)
        d = y1 - m * x1  # y intercept

        # quadratic formula to solve for x intersections
        a = 1 + m**2
        b = 2 * (-h + m * d - m * k)
        c = -2 * d * k + h**2 + d**2 + k**2 - r**2
        x_int[0] = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        x_int[1] = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)

        # solve for y intersections
        y_int[0] = m * x_int[0] + d
        y_int[1] = m * x_int[1] + d

        intersections = np.concatenate((x_int, y_int), axis=1)

    if np.all(intersections[0] == intersections[1]):
        exit_wound = intersections[0]

    else:
        p1_vec = (
            intersections[0] - intersections[1]
        )  # position vector with intersection point 1 for the direction and point 2 as the source
        p2_vec = (
            intersections[1] - intersections[0]
        )  # position vector with intersection point 2 for the direction and point 1 as the source
        p1_unit = p1_vec / norm(p1_vec)
        p2_unit = p2_vec / norm(p2_vec)

        angles = np.arccos(
            np.clip(np.dot(np.array([p1_unit, p2_unit]), ray_unit), -1.0, 1.0)
        )  # angles between vectors
        exit_wound = intersections[np.argmin(angles)]

    return exit_wound


def reflect_vector(
    point: np.ndarray, vector: np.ndarray, circle_center: np.ndarray
) -> np.ndarray:
    """
    This function reflects a vector off of a given point on a circle.
    Returns reflection.

    point: np.array([x, y])

    velocity: np.array([x, y])

    circle_center: np.array([x, y])

    reflection: np.array([x, y])
    """
    px, py = point
    vx, vy = vector
    cx, cy = circle_center

    if vx == vy == 0:
        return np.array([vx, vy])

    elif py == cy:
        reflection = np.array([-vx, vy])

    elif px == cx:
        reflection = np.array([vx, -vy])

    else:
        m_tangent = -(px - cx) / (
            py - cy
        )  # slope of the line that is tangent to circle at point

        m_rejection = (
            -1 / m_tangent
        )  # negative slope of rejection of vector on line that is tangent to circle at point
        b_rejection = vy - m_rejection * vx

        x_intersection = b_rejection / (m_tangent - m_rejection)
        y_intersection = m_tangent * x_intersection
        intersection = np.array(
            [x_intersection, y_intersection]
        )  # point at which vector rejection intersects with line that is tangent to circle at point

        rejection = intersection - vector  # negative of the rejection vector
        reflection = vector + 2 * rejection

    return reflection


def get_rot_angle(tail: np.ndarray, tip: np.ndarray):
    """
    This function returns the rotation angle (in radians)
    between the y-axis and the vector originating at tail
    and ending at tip.

    tail: np.array([x, y])

    tip: np.array([x, y])
    """
    y_prime = tip - tail
    y = np.array([0.0, 1.0])

    dot = np.dot(y, y_prime)
    theta = np.arccos(dot / (norm(y) * norm(y_prime)))

    cross = np.cross(y, y_prime)
    sign = np.sign(cross)

    if sign != 0.0:
        theta *= sign

    return theta


def generate_invisible_track_start(
    agent_pos,
    red_flag_pos,
    blue_flag_pos,
    agent_radius,
    keepout_radius,
    world_size,
    x_scrimmage,
    training_mode,
):
    flag_radius = keepout_radius - agent_radius

    assert (
        flag_radius >= agent_radius
    ), "Assumption that flag_keepout-agent_radius >= agent_radius is not satisfied"

    Circle(agent_pos, agent_radius)
    flag = Circle(red_flag_pos, flag_radius)

    agent2flag = Ray(agent_pos, red_flag_pos)

    agent_tangent_1, flag_tangent_1, agent_tangent_2, flag_tangent_2 = (
        get_interior_tangents(agent_pos, red_flag_pos, agent_radius, flag_radius)
    )

    flag_perpendicular = Line(flag_tangent_1, flag_tangent_2)  # divider

    ag2flag_edges1 = Ray(agent_tangent_1, flag_tangent_1)
    ag2flag_edges2 = Ray(agent_tangent_2, flag_tangent_2)

    ag2flag_edges_intersect = ag2flag_edges1.intersection(ag2flag_edges2)[0]

    int2pt1_dis = float(ag2flag_edges_intersect.distance(agent_pos))

    pt1 = ag2flag_edges_intersect + int2pt1_dis * np.asarray(
        agent2flag.direction.unit, dtype=float
    )
    pt1 = np.array(pt1, dtype=float)

    # environment corners (creating a slightly smaller environment than full size to calculate where track circle can go, moving all walls inwards by agent_radius)
    bottom_left = Point((agent_radius, agent_radius))
    bottom_right = Point((world_size[0] - agent_radius, agent_radius))
    top_left = Point((agent_radius, world_size[1] - agent_radius))
    top_right = Point((world_size[0] - agent_radius, world_size[1] - agent_radius))

    # walls
    top = Segment(top_left, top_right)
    bottom = Segment(bottom_left, bottom_right)
    left = Segment(bottom_left, top_left)
    right = Segment(bottom_right, top_right)

    # scrimmage line
    scrimmage_top = Point(x_scrimmage + agent_radius, world_size[1] - agent_radius)
    scrimmage_bottom = Point(x_scrimmage + agent_radius, agent_radius)
    scrimmage_line = Segment(scrimmage_top, scrimmage_bottom)

    walls = [top, bottom, left, right, scrimmage_line]
    scrimmage_line_idx = walls.index(scrimmage_line)

    # rays for triangular spawn area
    ray1 = Ray(pt1, pt1 + np.asarray(ag2flag_edges1.direction, dtype=float))
    ray2 = Ray(pt1, pt1 + np.asarray(ag2flag_edges2.direction, dtype=float))

    # ray-wall intersections
    ray1_wall_intersections = np.zeros(len(walls))
    ray2_wall_intersections = np.zeros(len(walls))
    for i in range(len(walls)):
        wall = walls[i]
        assert (
            0 <= len(ray1.intersection(wall)) <= 1
        ), "ray1 is intersecting with wall {} ({}) infinitely many times!}".format(
            i, wall
        )
        assert (
            0 <= len(ray2.intersection(wall)) <= 1
        ), "ray2 is intersecting with wall {} ({}) infinitely many times!}".format(
            i, wall
        )

        if not (i == scrimmage_line_idx and training_mode == "track"):
            ray1_wall_intersections[i] = len(ray1.intersection(wall)) > 0 and float(
                ray1.intersection(wall)[0][0]
            ) >= float(scrimmage_top.x)
            ray2_wall_intersections[i] = len(ray2.intersection(wall)) > 0 and float(
                ray2.intersection(wall)[0][0]
            ) >= float(scrimmage_top.x)

    # extend rays to their wall intersections
    # ray1
    if ray1_wall_intersections[scrimmage_line_idx] and pt1[0] > float(
        scrimmage_top.x
    ):
        # scrimmage line intersection
        pt2 = ray1.intersection(walls[scrimmage_line_idx])[0]  # pt2 generated by ray1

    elif ray1_wall_intersections[scrimmage_line_idx] and not pt1[0] > float(
        scrimmage_top.x
    ):
        ray1_wall_intersections[scrimmage_line_idx] = (
            False  # pt2 should not end at scrimmage line, change True to False
        )
        ray1_wall_intersection_idx = np.where(ray1_wall_intersections)[0][0]
        pt2 = ray1.intersection(walls[ray1_wall_intersection_idx])[
            0
        ]  # pt2 generated by ray1

    else:
        ray1_wall_intersection_idx = np.where(ray1_wall_intersections)[0][0]
        pt2 = ray1.intersection(walls[ray1_wall_intersection_idx])[
            0
        ]  # pt2 generated by ray1

    # ray2
    if ray2_wall_intersections[scrimmage_line_idx] and pt1[0] > float(
        scrimmage_top.x
    ):
        # scrimmage line intersection
        pt3 = ray2.intersection(walls[scrimmage_line_idx])[0]  # pt3 generated by ray2

    elif ray2_wall_intersections[scrimmage_line_idx] and not pt1[0] > float(
        scrimmage_top.x
    ):
        ray2_wall_intersections[scrimmage_line_idx] = (
            False  # pt3 should not end at scrimmage line, change True to False
        )
        ray2_wall_intersection_idx = np.where(ray2_wall_intersections)[0][0]
        pt3 = ray2.intersection(walls[ray2_wall_intersection_idx])[
            0
        ]  # pt3 generated by ray2

    else:
        ray2_wall_intersection_idx = np.where(ray2_wall_intersections)[0][0]
        pt3 = ray2.intersection(walls[ray2_wall_intersection_idx])[
            0
        ]  # pt3 generated by ray2

    pt2 = np.array(pt2, dtype=float)
    pt3 = np.array(pt3, dtype=float)

    # get pt4 and pt5 for the polygon
    a2f_divider_intersection = np.array(
        agent2flag.intersection(flag_perpendicular)[0], dtype=float
    )

    if norm(a2f_divider_intersection - agent_pos) > norm(pt1 - agent_pos):
        ray1_divider_intersection = ray1.intersection(flag_perpendicular)[0]
        ray2_divider_intersection = ray2.intersection(flag_perpendicular)[0]

        pt4 = np.array(
            ray1_divider_intersection, dtype=float
        )  # pt4 generated by ray1
        pt5 = np.array(
            ray2_divider_intersection, dtype=float
        )  # pt5 generated by ray2

        pt4_pt5_bool = True
        spawn_polygon_vert = 4  # base number of verticies for spawn area

    else:
        pt4_pt5_bool = False
        spawn_polygon_vert = 3  # base number of verticies for spawn area

    # checking if triangular region encompasses all of spawn region
    # get pt6 (corners and scrimmage top and bottom) if not encompassed
    # pt8 is corner not touched by rays, pt6 on border closer to ray1 intersection, pt7 on border closer to ray2 intersection

    encompassed = False
    for i in np.where(ray1_wall_intersections)[0]:
        for j in np.where(ray2_wall_intersections)[0]:
            if i == j:
                encompassed = True
                break
        if encompassed:
            break

    if not encompassed:
        # check up/ down
        up_a2f = False
        down_a2f = False
        if float(agent2flag.direction.y) > 0:
            up_a2f = True
        elif float(agent2flag.direction.y) < 0:
            down_a2f = True

        # check left/ right
        left_a2f = False
        right_a2f = False
        if float(agent2flag.direction.x) < 0:
            left_a2f = True
        elif float(agent2flag.direction.x) > 0:
            right_a2f = True

        pt6_bool = False
        pt7_bool = False
        pt8_bool = False

        if ray1_wall_intersections[scrimmage_line_idx]:
            # ray1 scrimmage (scrimmage intersection takes priority over others when intersecting from own side)
            if ray2_wall_intersections[walls.index(top)]:
                # ray2 hits top wall
                if up_a2f:
                    pt8 = np.array(scrimmage_top, dtype=float)
                    pt8_bool = True
                    spawn_polygon_vert += 1
                elif down_a2f:
                    pt8 = np.array(bottom_right, dtype=float)
                    pt8_bool = True
                    spawn_polygon_vert += 1

                    if not ray1_wall_intersections[walls.index(bottom)]:
                        # need a pt6 because ray1 does not intersect bottom of scrimmage line
                        pt6 = np.array(scrimmage_bottom, dtype=float)
                        pt6_bool = True
                        spawn_polygon_vert += 1
                    if not ray2_wall_intersections[walls.index(right)]:
                        # need a pt7 because ray2 does not intersect top right corner
                        pt7 = np.array(top_right, dtype=float)
                        pt7_bool = True
                        spawn_polygon_vert += 1

            elif ray2_wall_intersections[walls.index(bottom)]:
                # ray2 hits bottom wall
                if down_a2f:
                    pt8 = np.array(scrimmage_bottom, dtype=float)
                    pt8_bool = True
                    spawn_polygon_vert += 1
                elif up_a2f:
                    pt8 = np.array(top_right, dtype=float)
                    pt8_bool = True
                    spawn_polygon_vert += 1

                    if not ray1_wall_intersections[walls.index(top)]:
                        # need a pt6 because ray1 does not intersect top of scrimmage line
                        pt6 = np.array(scrimmage_top, dtype=float)
                        pt6_bool = True
                        spawn_polygon_vert += 1
                    if not ray2_wall_intersections[walls.index(right)]:
                        # need a pt7 because ray2 does not intersect bottom right corner
                        pt7 = np.array(bottom_right, dtype=float)
                        pt7_bool = True
                        spawn_polygon_vert += 1

            elif ray2_wall_intersections[walls.index(right)]:
                # ray2 hits right wall
                if up_a2f:
                    if not ray1_wall_intersections[walls.index(top)]:
                        # need a pt6 because ray1 does not intersect top of scrimmage line
                        pt6 = np.array(scrimmage_top, dtype=float)
                        pt6_bool = True
                        spawn_polygon_vert += 1
                    if not ray2_wall_intersections[walls.index(top)]:
                        # need a pt7 because ray2 does not intersect top right corner
                        pt7 = np.array(top_right, dtype=float)
                        pt7_bool = True
                        spawn_polygon_vert += 1
                elif down_a2f:
                    if not ray1_wall_intersections[walls.index(bottom)]:
                        # need a pt6 because ray1 does not intersect bottom of scrimmage line
                        pt6 = np.array(scrimmage_bottom, dtype=float)
                        pt6_bool = True
                        spawn_polygon_vert += 1
                    if not ray2_wall_intersections[walls.index(bottom)]:
                        # need a pt7 because ray2 does not intersect bottom right corner
                        pt7 = np.array(bottom_right, dtype=float)
                        pt7_bool = True
                        spawn_polygon_vert += 1

        elif ray2_wall_intersections[scrimmage_line_idx]:
            # ray2 scrimmage (scrimmage intersection takes priority over others when intersecting from own side)
            if ray1_wall_intersections[walls.index(top)]:
                # ray1 hits top wall
                if up_a2f:
                    pt8 = np.array(scrimmage_top, dtype=float)
                    pt8_bool = True
                    spawn_polygon_vert += 1
                elif down_a2f:
                    pt8 = np.array(bottom_right, dtype=float)
                    pt8_bool = True
                    spawn_polygon_vert += 1

                    if not ray2_wall_intersections[walls.index(bottom)]:
                        # need a pt7 because ray2 does not intersect bottom of scrimmage line
                        pt7 = np.array(scrimmage_bottom, dtype=float)
                        pt7_bool = True
                        spawn_polygon_vert += 1
                    if not ray1_wall_intersections[walls.index(right)]:
                        # need a pt6 because ray1 does not intersect top right corner
                        pt6 = np.array(top_right, dtype=float)
                        pt6_bool = True
                        spawn_polygon_vert += 1

            elif ray1_wall_intersections[walls.index(bottom)]:
                # ray1 hits bottom wall
                if down_a2f:
                    pt8 = np.array(scrimmage_bottom, dtype=float)
                    pt8_bool = True
                    spawn_polygon_vert += 1
                elif up_a2f:
                    pt8 = np.array(top_right, dtype=float)
                    pt8_bool = True
                    spawn_polygon_vert += 1

                    if not ray2_wall_intersections[walls.index(top)]:
                        # need a pt7 because ray2 does not intersect top of scrimmage line
                        pt7 = np.array(scrimmage_top, dtype=float)
                        pt7_bool = True
                        spawn_polygon_vert += 1
                    if not ray1_wall_intersections[walls.index(right)]:
                        # need a pt6 because ray1 does not intersect bottom right corner
                        pt6 = np.array(bottom_right, dtype=float)
                        pt6_bool = True
                        spawn_polygon_vert += 1

            elif ray1_wall_intersections[walls.index(right)]:
                # ray1 hits right wall
                if up_a2f:
                    if not ray2_wall_intersections[walls.index(top)]:
                        # need a pt7 because ray2 does not intersect top of scrimmage line
                        pt7 = np.array(scrimmage_top, dtype=float)
                        pt7_bool = True
                        spawn_polygon_vert += 1
                    if not ray1_wall_intersections[walls.index(top)]:
                        # need a pt6 because ray1 does not intersect top right corner
                        pt6 = np.array(top_right, dtype=float)
                        pt6_bool = True
                        spawn_polygon_vert += 1
                elif down_a2f:
                    if not ray2_wall_intersections[walls.index(bottom)]:
                        # need a pt7 because ray2 does not intersect bottom of scrimmage line
                        pt7 = np.array(scrimmage_bottom, dtype=float)
                        pt7_bool = True
                        spawn_polygon_vert += 1
                    if not ray1_wall_intersections[walls.index(bottom)]:
                        # need a pt6 because ray1 does not intersect bottom right corner
                        pt6 = np.array(bottom_right, dtype=float)
                        pt6_bool = True
                        spawn_polygon_vert += 1

        elif (
            ray1_wall_intersections[walls.index(bottom)]
            and ray2_wall_intersections[walls.index(top)]
        ) or (
            ray1_wall_intersections[walls.index(top)]
            and ray2_wall_intersections[walls.index(bottom)]
        ):
            # top and bottom
            if right_a2f:
                if ray1_wall_intersections[walls.index(top)]:
                    pt6 = np.array(top_right, dtype=float)
                    pt6_bool = True
                    spawn_polygon_vert += 1

                    pt7 = np.array(bottom_right, dtype=float)
                    pt7_bool = True
                    spawn_polygon_vert += 1

                elif ray1_wall_intersections[walls.index(bottom)]:
                    pt6 = np.array(bottom_right, dtype=float)
                    pt6_bool = True
                    spawn_polygon_vert += 1

                    pt7 = np.array(top_right, dtype=float)
                    pt7_bool = True
                    spawn_polygon_vert += 1

            elif left_a2f:
                if training_mode == "target2":
                    if ray1_wall_intersections[walls.index(top)]:
                        pt6 = np.array(scrimmage_top, dtype=float)
                        pt6_bool = True
                        spawn_polygon_vert += 1

                        pt7 = np.array(scrimmage_bottom, dtype=float)
                        pt7_bool = True
                        spawn_polygon_vert += 1

                    elif ray1_wall_intersections[walls.index(bottom)]:
                        pt6 = np.array(scrimmage_bottom, dtype=float)
                        pt6_bool = True
                        spawn_polygon_vert += 1

                        pt7 = np.array(scrimmage_top, dtype=float)
                        pt7_bool = True
                        spawn_polygon_vert += 1

                elif training_mode == "track":
                    if (
                        ray1_wall_intersections[walls.index(top)]
                        and not ray1_wall_intersections[walls.index(left)]
                    ):
                        pt6 = np.array(top_left, dtype=float)
                        pt6_bool = True
                        spawn_polygon_vert += 1

                        if not ray2_wall_intersections[walls.index(left)]:
                            pt7 = np.array(bottom_left, dtype=float)
                            pt7_bool = True
                            spawn_polygon_vert += 1

                    elif (
                        ray1_wall_intersections[walls.index(bottom)]
                        and not ray1_wall_intersections[walls.index(left)]
                    ):
                        pt6 = np.array(bottom_left, dtype=float)
                        pt6_bool = True
                        spawn_polygon_vert += 1

                        if not ray2_wall_intersections[walls.index(left)]:
                            pt7 = np.array(top_left, dtype=float)
                            pt7_bool = True
                            spawn_polygon_vert += 1

        elif (
            ray1_wall_intersections[walls.index(top)]
            and ray2_wall_intersections[walls.index(right)]
        ) or (
            ray1_wall_intersections[walls.index(right)]
            and ray2_wall_intersections[walls.index(top)]
        ):
            # top and right

            if up_a2f:
                pt8 = np.array(top_right, dtype=float)
                pt8_bool = True
                spawn_polygon_vert += 1

            elif down_a2f:
                if training_mode == "target2":
                    pt8 = np.array(scrimmage_bottom, dtype=float)
                    pt8_bool = True
                    spawn_polygon_vert += 1

                    if ray1_wall_intersections[walls.index(top)]:
                        pt6 = np.array(scrimmage_top, dtype=float)
                        pt6_bool = True
                        spawn_polygon_vert += 1

                        if not ray2_wall_intersections[walls.index(bottom)]:
                            pt7 = np.array(bottom_right, dtype=float)
                            pt7_bool = True
                            spawn_polygon_vert += 1

                    elif ray1_wall_intersections[walls.index(right)]:
                        if not ray1_wall_intersections[walls.index(bottom)]:
                            pt6 = np.array(bottom_right, dtype=float)
                            pt6_bool = True
                            spawn_polygon_vert += 1

                        pt7 = np.array(scrimmage_top, dtype=float)
                        pt7_bool = True
                        spawn_polygon_vert += 1

                elif training_mode == "track":
                    pt8 = np.array(bottom_left, dtype=float)
                    pt8_bool = True
                    spawn_polygon_vert += 1

                    if ray1_wall_intersections[walls.index(top)]:
                        if not ray1_wall_intersections[walls.index(left)]:
                            pt6 = np.array(top_left, dtype=float)
                            pt6_bool = True
                            spawn_polygon_vert += 1

                        if not ray2_wall_intersections[walls.index(bottom)]:
                            pt7 = np.array(bottom_right, dtype=float)
                            pt7_bool = True
                            spawn_polygon_vert += 1

                    elif ray1_wall_intersections[walls.index(right)]:
                        if not ray1_wall_intersections[walls.index(bottom)]:
                            pt6 = np.array(bottom_right, dtype=float)
                            pt6_bool = True
                            spawn_polygon_vert += 1

                        if not ray2_wall_intersections[walls.index(left)]:
                            pt7 = np.array(top_left, dtype=float)
                            pt7_bool = True
                            spawn_polygon_vert += 1

        elif (
            ray1_wall_intersections[walls.index(bottom)]
            and ray2_wall_intersections[walls.index(right)]
        ) or (
            ray1_wall_intersections[walls.index(right)]
            and ray2_wall_intersections[walls.index(bottom)]
        ):
            # bottom and right

            if down_a2f:
                pt8 = np.array(bottom_right, dtype=float)
                pt8_bool = True
                spawn_polygon_vert += 1

            elif up_a2f:
                if training_mode == "target2":
                    pt8 = np.array(scrimmage_top, dtype=float)
                    pt8_bool = True
                    spawn_polygon_vert += 1

                    if ray1_wall_intersections[walls.index(bottom)]:
                        pt6 = np.array(scrimmage_bottom, dtype=float)
                        pt6_bool = True
                        spawn_polygon_vert += 1

                        if not ray2_wall_intersections[walls.index(top)]:
                            pt7 = np.array(top_right, dtype=float)
                            pt7_bool = True
                            spawn_polygon_vert += 1

                    elif ray1_wall_intersections[walls.index(right)]:
                        if not ray1_wall_intersections[walls.index(top)]:
                            pt6 = np.array(top_right, dtype=float)
                            pt6_bool = True
                            spawn_polygon_vert += 1

                        pt7 = np.array(scrimmage_bottom, dtype=float)
                        pt7_bool = True
                        spawn_polygon_vert += 1

                elif training_mode == "track":
                    pt8 = np.array(top_left, dtype=float)
                    pt8_bool = True
                    spawn_polygon_vert += 1

                    if ray1_wall_intersections[walls.index(bottom)]:
                        if not ray1_wall_intersections[walls.index(left)]:
                            pt6 = np.array(bottom_left, dtype=float)
                            pt6_bool = True
                            spawn_polygon_vert += 1

                        if not ray2_wall_intersections[walls.index(top)]:
                            pt7 = np.array(top_right, dtype=float)
                            pt7_bool = True
                            spawn_polygon_vert += 1

                    elif ray1_wall_intersections[walls.index(right)]:
                        if not ray1_wall_intersections[walls.index(top)]:
                            pt6 = np.array(top_right, dtype=float)
                            pt6_bool = True
                            spawn_polygon_vert += 1

                        if not ray2_wall_intersections[walls.index(left)]:
                            pt7 = np.array(bottom_left, dtype=float)
                            pt7_bool = True
                            spawn_polygon_vert += 1

        elif (
            ray1_wall_intersections[walls.index(left)]
            and ray2_wall_intersections[walls.index(right)]
        ) or (
            ray1_wall_intersections[walls.index(right)]
            and ray2_wall_intersections[walls.index(left)]
        ):
            # left and right

            if up_a2f:
                if (
                    ray1_wall_intersections[walls.index(left)]
                    and not ray1_wall_intersections[walls.index(top)]
                ):
                    pt6 = np.array(top_left, dtype=float)
                    pt6_bool = True
                    spawn_polygon_vert += 1

                    if not ray2_wall_intersections[walls.index(top)]:
                        pt7 = np.array(top_right, dtype=float)
                        pt7_bool = True
                        spawn_polygon_vert += 1

                elif (
                    ray1_wall_intersections[walls.index(right)]
                    and not ray1_wall_intersections[walls.index(top)]
                ):
                    pt6 = np.array(top_right, dtype=float)
                    pt6_bool = True
                    spawn_polygon_vert += 1

                    if not ray2_wall_intersections[walls.index(top)]:
                        pt7 = np.array(top_left, dtype=float)
                        pt7_bool = True
                        spawn_polygon_vert += 1

            elif down_a2f:
                if (
                    ray1_wall_intersections[walls.index(left)]
                    and not ray1_wall_intersections[walls.index(bottom)]
                ):
                    pt6 = np.array(bottom_left, dtype=float)
                    pt6_bool = True
                    spawn_polygon_vert += 1

                    if not ray2_wall_intersections[walls.index(bottom)]:
                        pt7 = np.array(bottom_right, dtype=float)
                        pt7_bool = True
                        spawn_polygon_vert += 1

                elif (
                    ray1_wall_intersections[walls.index(right)]
                    and not ray1_wall_intersections[walls.index(bottom)]
                ):
                    pt6 = np.array(bottom_right, dtype=float)
                    pt6_bool = True
                    spawn_polygon_vert += 1

                    if not ray2_wall_intersections[walls.index(bottom)]:
                        pt7 = np.array(bottom_left, dtype=float)
                        pt7_bool = True
                        spawn_polygon_vert += 1

        if training_mode == "track":
            # scenarios that can only happen in track mode

            if (
                ray1_wall_intersections[walls.index(top)]
                and ray2_wall_intersections[walls.index(left)]
            ) or (
                ray1_wall_intersections[walls.index(left)]
                and ray2_wall_intersections[walls.index(top)]
            ):
                # top and left

                if up_a2f:
                    pt8 = np.array(top_left, dtype=float)
                    pt8_bool = True
                    spawn_polygon_vert += 1

                elif down_a2f:
                    pt8 = np.array(bottom_right, dtype=float)
                    pt8_bool = True
                    spawn_polygon_vert += 1

                    if (
                        ray1_wall_intersections[walls.index(top)]
                        and not ray1_wall_intersections[walls.index(right)]
                    ):
                        pt6 = np.array(top_right, dtype=float)
                        pt6_bool = True
                        spawn_polygon_vert += 1

                        if not ray2_wall_intersections[walls.index(bottom)]:
                            pt7 = np.array(bottom_left, dtype=float)
                            pt7_bool = True
                            spawn_polygon_vert += 1

                    elif (
                        ray1_wall_intersections[walls.index(left)]
                        and not ray1_wall_intersections[walls.index(bottom)]
                    ):
                        pt6 = np.array(bottom_left, dtype=float)
                        pt6_bool = True
                        spawn_polygon_vert += 1

                        if not ray2_wall_intersections[walls.index(right)]:
                            pt7 = np.array(top_right, dtype=float)
                            pt7_bool = True
                            spawn_polygon_vert += 1

            elif (
                ray1_wall_intersections[walls.index(bottom)]
                and ray2_wall_intersections[walls.index(left)]
            ) or (
                ray1_wall_intersections[walls.index(left)]
                and ray2_wall_intersections[walls.index(bottom)]
            ):
                # bottom and left

                if down_a2f:
                    pt8 = np.array(bottom_left, dtype=float)
                    pt8_bool = True
                    spawn_polygon_vert += 1

                elif up_a2f:
                    pt8 = np.array(top_right, dtype=float)
                    pt8_bool = True
                    spawn_polygon_vert += 1

                    if (
                        ray1_wall_intersections[walls.index(bottom)]
                        and not ray1_wall_intersections[walls.index(right)]
                    ):
                        pt6 = np.array(bottom_right, dtype=float)
                        pt6_bool = True
                        spawn_polygon_vert += 1

                        if not ray2_wall_intersections[walls.index(top)]:
                            pt7 = np.array(top_left, dtype=float)
                            pt7_bool = True
                            spawn_polygon_vert += 1

                    elif (
                        ray1_wall_intersections[walls.index(left)]
                        and not ray1_wall_intersections[walls.index(top)]
                    ):
                        pt6 = np.array(top_left, dtype=float)
                        pt6_bool = True
                        spawn_polygon_vert += 1

                        if not ray2_wall_intersections[walls.index(right)]:
                            pt7 = np.array(bottom_right, dtype=float)
                            pt7_bool = True
                            spawn_polygon_vert += 1

    # make spawn triangles
    if spawn_polygon_vert == 3:
        spawn_triangles = [Triangle(pt1, pt2, pt3)]

    elif spawn_polygon_vert == 4:
        if pt4_pt5_bool:
            spawn_triangles = [Triangle(pt4, pt2, pt5), Triangle(pt5, pt3, pt2)]
        else:
            if pt6_bool:
                # pt6
                spawn_triangles = [Triangle(pt1, pt2, pt3), Triangle(pt2, pt6, pt3)]
            elif pt7_bool:
                # pt7
                spawn_triangles = [Triangle(pt1, pt2, pt3), Triangle(pt2, pt7, pt3)]
            elif pt8_bool:
                # pt8
                spawn_triangles = [Triangle(pt1, pt2, pt3), Triangle(pt2, pt8, pt3)]

    elif spawn_polygon_vert == 5:
        if pt4_pt5_bool:
            if pt6_bool:
                # pt6
                spawn_triangles = [
                    Triangle(pt4, pt2, pt5),
                    Triangle(pt5, pt3, pt2),
                    Triangle(pt2, pt3, pt6),
                ]
            elif pt7_bool:
                # pt7
                spawn_triangles = [
                    Triangle(pt4, pt2, pt5),
                    Triangle(pt5, pt3, pt2),
                    Triangle(pt2, pt3, pt7),
                ]
            elif pt8_bool:
                # pt8
                spawn_triangles = [
                    Triangle(pt4, pt2, pt5),
                    Triangle(pt5, pt3, pt2),
                    Triangle(pt2, pt3, pt8),
                ]
        else:
            if pt6_bool and pt7_bool:
                # pt6 & pt7
                spawn_triangles = [
                    Triangle(pt1, pt2, pt3),
                    Triangle(pt2, pt6, pt3),
                    Triangle(pt6, pt7, pt3),
                ]
            elif pt6_bool and pt8_bool:
                # pt6 & pt8
                spawn_triangles = [
                    Triangle(pt1, pt2, pt3),
                    Triangle(pt2, pt6, pt3),
                    Triangle(pt6, pt8, pt3),
                ]
            elif pt7_bool and pt8_bool:
                # pt7 & pt8
                spawn_triangles = [
                    Triangle(pt1, pt2, pt3),
                    Triangle(pt2, pt8, pt3),
                    Triangle(pt8, pt7, pt3),
                ]

    elif spawn_polygon_vert == 6:
        if pt4_pt5_bool:
            if pt6_bool and pt7_bool:
                # pt6 & pt7
                spawn_triangles = [
                    Triangle(pt4, pt2, pt5),
                    Triangle(pt5, pt3, pt2),
                    Triangle(pt2, pt6, pt3),
                    Triangle(pt3, pt7, pt6),
                ]
            elif pt6_bool and pt8_bool:
                # pt6 & pt8
                spawn_triangles = [
                    Triangle(pt4, pt2, pt5),
                    Triangle(pt5, pt3, pt2),
                    Triangle(pt2, pt6, pt3),
                    Triangle(pt3, pt8, pt6),
                ]
            elif pt7_bool and pt8_bool:
                # pt7 & pt8
                spawn_triangles = [
                    Triangle(pt4, pt2, pt5),
                    Triangle(pt5, pt3, pt2),
                    Triangle(pt2, pt8, pt3),
                    Triangle(pt3, pt7, pt8),
                ]
        else:
            # pt6, pt7, pt8
            spawn_triangles = [
                Triangle(pt1, pt2, pt3),
                Triangle(pt2, pt6, pt3),
                Triangle(pt6, pt7, pt3),
                Triangle(pt6, pt8, pt7),
            ]

    elif spawn_polygon_vert == 7:
        # pt6, pt7, pt8
        spawn_triangles = [
            Triangle(pt4, pt2, pt5),
            Triangle(pt5, pt3, pt2),
            Triangle(pt2, pt6, pt3),
            Triangle(pt3, pt7, pt6),
            Triangle(pt6, pt7, pt8),
        ]

    # generate selection probability for each spawn triangles
    triangle_spawn_areas = np.zeros(len(spawn_triangles))

    for t in range(len(spawn_triangles)):
        triangle_spawn_areas[t] = np.abs(float(spawn_triangles[t].area))

    total_area = np.sum(triangle_spawn_areas)
    triangle_spawn_probability = triangle_spawn_areas / total_area

    # generate track point and check that not too far inside of flag
    # set flag collision radius
    if training_mode == "target2":
        red_flag_collision_radius = flag_radius
        blue_flag_collision_radius = keepout_radius
    elif training_mode == "track":
        red_flag_collision_radius = keepout_radius + 2 * agent_radius
        blue_flag_collision_radius = keepout_radius  # only need this in track training mode (blue won't go on its own side in target2)

    collision_red_flag = True
    collision_blue_flag = True
    while collision_red_flag or collision_blue_flag:
        # choose a spawn triangle (st)
        st = np.random.choice(spawn_triangles, p=triangle_spawn_probability)
        track_pos = point_on_triangle(st)

        # check red flag distance
        red_flag_dis = float(flag.center.distance(track_pos))

        if red_flag_dis >= red_flag_collision_radius:
            collision_red_flag = False
        else:
            collision_red_flag = True
            continue

        # check blue flag distance
        blue_flag_dis = np.linalg.norm(track_pos - blue_flag_pos)

        if blue_flag_dis > blue_flag_collision_radius:
            collision_blue_flag = False
        else:
            collision_blue_flag = True
            continue

    return track_pos


def point_on_triangle(triangle):
    """
    Random point on the triangle with vertices pt1, pt2 and pt3.

    triangle: sympy Triangle object
    """
    pt1 = np.array(triangle.vertices[0], dtype=float)
    pt2 = np.array(triangle.vertices[1], dtype=float)
    pt3 = np.array(triangle.vertices[2], dtype=float)

    x, y = sorted([np.random.rand(), np.random.rand()])
    s, t, u = x, y - x, 1 - y
    return np.array(
        [s * pt1[0] + t * pt2[0] + u * pt3[0], s * pt1[1] + t * pt2[1] + u * pt3[1]]
    )


def get_interior_tangents(c1, c2, r1, r2):
    """
    Assumes that there are two interior tangents.

    c1 and c2 are coordinates of circle centers
    """
    hypotenuse = np.linalg.norm(c1 - c2)
    short = r1 + r2

    phi1 = (
        np.arctan2(c2[1] - c1[1], c2[0] - c1[0])
        + np.arcsin(short / hypotenuse)
        - np.pi / 2
    )
    t1 = np.array([c1[0] + r1 * np.cos(phi1), c1[1] + r1 * np.sin(phi1)])
    t2 = np.array(
        [c2[0] + r2 * np.cos(phi1 + np.pi), c2[1] + r2 * np.sin(phi1 + np.pi)]
    )

    phi2 = (
        np.arctan2(c2[1] - c1[1], c2[0] - c1[0])
        - np.arcsin(short / hypotenuse)
        + np.pi / 2
    )
    s1 = np.array([c1[0] + r1 * np.cos(phi2), c1[1] + r1 * np.sin(phi2)])
    s2 = np.array(
        [c2[0] + r2 * np.cos(phi2 + np.pi), c2[1] + r2 * np.sin(phi2 + np.pi)]
    )

    return t1, t2, s1, s2


def get_screen_res():
    """Returns the screen resolution as [Width, Height]."""
    pygame.init()
    screen_info = pygame.display.Info()
    res = [screen_info.current_w, screen_info.current_h]
    pygame.quit()
    return res


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception("lines do not intersect")

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return np.array([x, y])


def closest_point_on_line(A, B, P):
    """
    Args:
        A: an (x, y) point on a line
        B: a different (x, y) point on a line
        P: the point to minimize the distance from.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    P = np.asarray(P)

    v_AB = B - A
    v_AP = P - A
    len_AB = np.linalg.norm(v_AB)
    unit_AB = v_AB / len_AB
    v_AB_AP = np.dot(v_AP, v_AB)
    proj_dist = np.divide(v_AB_AP, len_AB)

    if proj_dist <= 0.0:
        return A
    elif proj_dist >= len_AB:
        return B
    else:
        return [A[0] + (unit_AB[0] * proj_dist), A[1] + (unit_AB[1] * proj_dist)]


def vector_to(A, B, unit=False):
    """
    Returns a vector from A to B
    if unit is true, will scale the vector.
    """
    A = np.asarray(A)
    B = np.asarray(B)

    vec = B - A

    if unit:
        norm = np.linalg.norm(vec)
        if norm >= 1e-5:
            vec = vec / norm

    return vec


def heading_angle_conversion(deg):
    """
    Converts a world-frame angle to a heading and vice-versa
    The transformation is its own inverse
    Args:
        deg: the angle (heading) in degrees
    Returns:
        float: the heading (angle) in degrees.
    """
    return (90 - deg) % 360


def vec_to_mag_heading(vec):
    """Converts a vector to a magnitude and heading (deg)."""
    mag = np.linalg.norm(vec)
    angle = math.degrees(math.atan2(vec[1], vec[0]))
    return mag, angle180(heading_angle_conversion(angle))


def mag_heading_to_vec(mag, bearing):
    """Converts a magnitude and heading (deg) to a vector."""
    angle = math.radians(heading_angle_conversion(bearing))
    x = mag * math.cos(angle)
    y = mag * math.sin(angle)
    return np.array([x, y], dtype=np.float32)


def mag_bearing_to(A, B, relative_hdg=None):
    """
    Returns magnitude and bearing vector between two points.

    if relative_hdg provided then return bearing relative to that

    Returns
    -------
        float: distance
        float: bearing in [-180, 180] which is relative
               to relative_hdg if provided and otherwise
               is global
    """
    mag, hdg = vec_to_mag_heading(vector_to(A, B))
    if relative_hdg is not None:
        hdg = (hdg - relative_hdg) % 360
    return mag, angle180(hdg)


def angle180(deg):
    while deg > 180:
        deg -= 360
    while deg < -180:
        deg += 360
    return deg


def clip(val, minimum, maximum):
    if val > maximum:
        return maximum
    elif val < minimum:
        return minimum
    else:
        return val
