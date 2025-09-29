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

import numpy as np


class PID:
    """
    Simple class for scalar PID control.
    
    Adapted from MOOS-IvP ScalarPID and PIDEngine:
        (1) https://oceanai.mit.edu/svn/moos-ivp-aro/trunk/ivp/src/lib_marine_pid/ScalarPID.cpp
            -deriv_history_size (m_nHistorySize) default changed from 10 to 1 (moving average for smoothing not necessary in sim)

        (2) https://oceanai.mit.edu/svn/moos-ivp-aro/trunk/ivp/src/lib_marine_pid/PIDEngine.cpp
    """
    def __init__(self, dt, kp, kd, ki, deriv_history_size=1, integral_limit=float("inf"), output_limit=float("inf"), n_envs=1):
        self._dt = dt
        self._kp = kp
        self._kd = kd
        self._ki = ki

        self._deriv_history = np.full((n_envs, deriv_history_size), np.nan)
        self._integral_limit = integral_limit
        self._output_limit = output_limit

        self._prev_error = np.full(n_envs, np.nan)
        self._integral = np.zeros(n_envs)

    def __call__(self, error, env_idxs):
        #Calculate the derivative term
        self._deriv_history[env_idxs] = np.roll(self._deriv_history[env_idxs], 1, axis=-1)
        self._deriv_history[env_idxs, 0] = np.where(
            np.isnan(self._prev_error[env_idxs]),
            self._deriv_history[env_idxs, 0],
            (error - self._prev_error[env_idxs]) / self._dt
        )
        deriv = np.zeros(env_idxs.shape[0])
        has_prev_error = ~np.isnan(self._prev_error[env_idxs])
        deriv[has_prev_error] = np.nanmean(self._deriv_history[env_idxs][has_prev_error], axis=-1)

        #Calculate the integral term
        self._integral[env_idxs] += self._ki * error * self._dt
        np.clip(self._integral[env_idxs], -self._integral_limit, self._integral_limit, out=self._integral[env_idxs]) #prevent integral wind up
        
        #Calculate PID output
        pid_out = (self._kp * error) + (self._kd * deriv) + self._integral #note Ki is already in self._integral
        np.clip(pid_out, -self._output_limit, self._output_limit, out=pid_out) #prevent saturation

        self._prev_error[env_idxs] = error

        return pid_out
