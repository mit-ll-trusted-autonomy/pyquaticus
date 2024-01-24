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

class PID:
    """Simple class for scalar PID control."""

    def __init__(self, dt, kp, ki, kd, integral_max=float("inf")):
        self._dt = dt
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._integral_max = integral_max

        self._prev_error = 0.0
        self._integral = 0.0

    def __call__(self, error):
        self._integral = min(self._integral + error * self._dt, self._integral_max)
        deriv = (error - self._prev_error) / self._dt

        self._prev_error = error

        p = self._kp * error
        i = self._ki * self._integral
        d = self._kd * deriv

        return p + i + d
