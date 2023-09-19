import math
import numpy as np

#----------------------------------------------------------------------
# Auxiliary classes
class PNT:
    """Encapsulate position, navigation, and timing (PNT) information for a ship"""
    def __init__(self, name='unknown', tagging_cooldown=10.0):
        self.name = name
        self.x = 0.0
        self.y = 0.0
        self.speed = 0.0
        self.heading = 0.0
        self.tagged = False
        # can tag when it counts up to the max again (see config for max)
        self.tagging_cooldown = tagging_cooldown

    def __repr__(self):
        return (f'{self.name}: ({self.x:.2f}, {self.y:.2f}) '
                + f'{self.speed:.1f} m/s {self.heading:.0f} deg '
                + f'tagged={self.tagged} '
                + f'tagging_cooldown={self.tagging_cooldown}')

class Vertex:
    def	__init__(self, x=0.0, y=None):
        """Simple (x,y) point"""
        if y is	None:
            if isinstance(x, list) or isinstance(x, tuple) or isinstance(x, np.ndarray):
                self.x = x[0]
                self.y = x[1]
            else:
                self.x = x
                self.y = x
        else:
            self.x = x
            self.y = y

    def __repr__(self):
        return(f'({self.x}, {self.y})')


def heading_angle_conversion(deg):
    '''
    Converts a world-frame angle to a heading and vice-versa
    The transformation is its own inverse

    Args:
        deg: the angle (heading) in degrees

    Returns:
        float: the heading (angle) in degrees
    '''

    return (90 - deg) % 360


class AxisAligner:
    def __init__(self, desired_pnts):
        '''
        This class can be used to rotate an Aquaticus CTF
        field so that it is axis-aligned. It does this by
        ensuring that the scrimmage line is perfectly vertical.

        This was designed for the West Point location at Lake Popolopen

        Note: not all the fields are perfectly rectangular, so it's
        not guaranteed that the other field edges are perfectly aligned.

        Args:
            desired_pnts: a dictionary with the points corresponding to
                          the following (after rotation)
                          i.e., these are the desired roles of these points

                        ll: lower left point before transformation
                        ul: upper left point before transformation
                        lr: lower right point before transformation
                        ur: upper right point before transformation
                        ls: lower scrimmage point before transformation
                        us: upper scrimmage point before transformation
        '''
        for key in desired_pnts:
            desired_pnts[key] = np.asarray(desired_pnts[key], dtype=np.float32)
        # just treating it as a rectangle even thought it's actually not
        self.lower_left_corner = desired_pnts['ll']
        scrimmage_lower = desired_pnts['ls']
        scrimmage_upper = desired_pnts['us']
        vec = scrimmage_upper - scrimmage_lower
        # rotate so that scrimmage line is aligned with y-axis
        angle = math.atan2(vec[1], vec[0]) - math.pi/2
        self.rot_matrix = np.array([[math.cos(angle), -math.sin(angle)],
                                    [math.sin(angle), math.cos(angle)]],
                                   dtype=np.float32)
        self.axis_angle = math.degrees(angle)
        self.scrimmage = self.to_axis_aligned_coord(desired_pnts['ls'], translate=True)[0][0]

        ll, ul, lr, ur = self.to_axis_aligned_coord(np.stack(
                                                (desired_pnts['ll'],
                                                desired_pnts['ul'],
                                                desired_pnts['lr'],
                                                desired_pnts['ur'])),
                                            translate=True)
        # give conservative world bounds
        # conservative because field is not true rectangle
        # giving boundaries of an inscribed rectangle
        self.x_min = max(ll[0], ul[0])
        self.y_min = max(ll[1], lr[1])
        self.x_max = min(lr[0], ur[0])
        self.y_max = min(ul[1], ur[1])


    def to_axis_aligned_coord(self, pnts, translate=False):
        '''
        Transforms points in the absolute (MOOS) coordinate system
        to one where the lower left corner of the field is (0, 0)
        and is rotated so that the bottom of the field is aligned
        with the x-axis

        Args:
            pnts: Nx2 numpy array of points
            translate: flag on whether to translate the points

        Returns:
            np.ndarray: Nx2 transformed points
        '''
        pnts = pnts.reshape((-1, 2))
        assert pnts.shape[1] == 2
        if translate:
            pnts = pnts - self.lower_left_corner
        return np.matmul(pnts, self.rot_matrix)

    def to_absolute_coord(self, pnts, translate=False):
        '''
        Transforms points in our axis-aligned coordinates to the
        absolute (MOOS) coordinates.

        Args:
            pnts: Nx2 numpy array of points

        Returns:
            np.ndarray: Nx2 transformed points
        '''
        assert pnts.shape[1] == 2
        rotated_points = np.matmul(pnts, self.rot_matrix.T)
        if translate:
            rotated_points + self.lower_left_corner
        return rotated_points.T

    def to_axis_aligned_heading(self, hdg):
        return (hdg+self.axis_angle)%360

    def to_absolute_heading(self, hdg):
        return (hdg-self.axis_angle)%360

# Adapted from ctf-gym
def closest_point_on_line(A,B,P):
    '''
    Args:
        A: an (x, y) point on a line
        B: a different (x, y) point on a line
        P: the point to minimize the distance from
    '''
    A = np.asarray(A)
    B = np.asarray(B)
    P = np.asarray(P)

    v_AB = B-A
    v_AP = P-A
    len_AB = np.linalg.norm(v_AB)
    unit_AB = v_AB/len_AB
    v_AB_AP = np.dot(v_AP,v_AB)
    proj_dist = np.divide(v_AB_AP,len_AB)

    if proj_dist <= 0.0:
        return A
    elif proj_dist >= len_AB:
        return B
    else:
        return [A[0] + (unit_AB[0] * proj_dist),A[1] + (unit_AB[1] * proj_dist)]

def vector_to(A, B, unit=False):
    '''
    Returns a vector from A to B
    if unit is true, will scale the vector
    '''
    A = np.asarray(A)
    B = np.asarray(B)

    vec = B - A

    if unit:
        norm = np.linalg.norm(vec)
        if norm >= 1e-5:
            vec = vec / norm

    return vec

def mag_heading_to(A, B, relative_hdg=None):
    '''
    Returns magnitude and heading vector between two points

    if relative_hdg provided then return heading relative to that

    Returns:
        float: distance
        float: heading in [-180, 180] which is relative
               to relative_hdg if provided and otherwise
               is global
    '''
    mag, hdg = vec_to_mag_heading(vector_to(A, B))
    if relative_hdg is not None:
        hdg = (hdg - relative_hdg) % 360
    return mag, zero_centered_heading(hdg)

def vec_to_mag_heading(vec):
    mag = np.linalg.norm(vec)
    angle = math.degrees(math.atan2(vec[1], vec[0]))
    return mag, heading_angle_conversion(angle)

def mag_heading_to_vec(mag, heading, unit=False):
    angle = math.radians(heading_angle_conversion(heading))
    x = mag*math.cos(angle)
    y = mag*math.sin(angle)
    res = np.array([x, y], dtype=np.float32)
    if unit:
        res_norm = np.linalg.norm(res)
        if res_norm > 1e-5:
            res = res / res_norm
    return res

def zero_centered_heading(hdg):
    '''
    Transforms heading from [0, 360] to [-180, 180]
    '''
    if hdg > 180:
        return hdg - 360
    else:
        return hdg


class ArtificialPotentialField:
    def __init__(self, dt, threshold, a, b=None):
        self._dt = dt
        self._threshold = threshold
        self._a = a
        self._b = b
        if b is not None:
            dy = self._b[1] - self._a[1]
            dx = self._b[0] - self._a[0]
            self._normal = np.array([-dy, dx], dtype=np.float32)
            self._normal = self._normal / np.linalg.norm(self._normal)
            self._force_angle = math.atan2(self._normal[1], self._normal[0])

    def __call__(self, pnt, action_vec):
        if self._b is None:
            # point case
            pos = np.array([pnt.x, pnt.y], dtype=np.float32)
            avoid_pnt = np.array(self._a, dtype=np.float32)
            dist, hdg = mag_heading_to(pos, avoid_pnt)
            if dist > self._threshold:
                return np.array([0., 0.], dtype=np.float32)
            else:
                # get speed component along the unsafe direction
                mvmt_vec = mag_heading_to_vec(pnt.speed, pnt.heading)
                avoid_to_pos = vector_to(avoid_pnt, pos, unit=True)
                average_vec = action_vec + mvmt_vec
                comp = np.dot(average_vec, -avoid_to_pos)
                if comp < 1e-2:
                    return np.array([0., 0.], dtype=np.float32)

                # 10% boost in force to ensure you don't cross over
                force_mag = 1.1 * comp * self._threshold / (1e-2 + dist)
                edit = force_mag*avoid_to_pos
                desired_action = average_vec + edit
                return desired_action - action_vec
        else:
            # line case
            pos = np.array([pnt.x, pnt.y], dtype=np.float32)
            cp = closest_point_on_line(self._a, self._b, pos)
            dist, hdg = mag_heading_to(pos, cp)
            if dist > self._threshold:
                return np.array([0., 0.], dtype=np.float32)
            else:
                # get speed component along the normal direction
                mvmt_vec = mag_heading_to_vec(pnt.speed, pnt.heading)
                average_vec = action_vec + mvmt_vec
                comp = np.dot(average_vec, -self._normal)
                if comp < 1e-2:
                    return np.array([0., 0.], dtype=np.float32)

                # 5% boost in force to ensure you don't cross over
                force_mag = 1.05 * comp * self._threshold / (1e-2 + dist)
                edit = force_mag*self._normal
                desired_action = average_vec + edit
                return desired_action - action_vec