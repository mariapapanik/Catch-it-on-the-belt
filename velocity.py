#!/usr/bin/env python3
import time
import numpy as np
from collections import deque
import config

class BeltVelocityEstimator:
    def __init__(self, window=config.TRACK_WINDOW):
        self._t    = deque(maxlen=window)
        self._vals = deque(maxlen=window)
        self.last  = {'x': None, 'y': None, 'z': None}

    def update(self, rx, ry, rz):
        self._t.append(time.monotonic())
        val = {'x': rx, 'y': ry, 'z': rz}[config.BELT_AXIS]
        self._vals.append(val)
        self.last = {'x': rx, 'y': ry, 'z': rz}

    def reset(self):
        self._t.clear()
        self._vals.clear()
        self.last = {'x': None, 'y': None, 'z': None}

    def velocity_and_predict(self, target_val):
        if len(self._t) < config.MIN_SAMPLES_FIT:
            return None, None
        t_arr = np.array(self._t) - self._t[0]
        v_arr = np.array(self._vals)
        A      = np.column_stack([t_arr, np.ones_like(t_arr)])
        result = np.linalg.lstsq(A, v_arr, rcond=None)
        v, b   = result[0]
        if abs(v) < 1e-4:
            return 0.0, None
        t_now   = time.monotonic() - self._t[0]
        val_now = v * t_now + b
        eta     = (target_val - val_now) / v
        return v, eta
