from __future__ import annotations

import numpy as np


class TorchCompat:
    long = np.int64
    int = np.int64

    @staticmethod
    def zeros(shape, dtype=int):
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def tensor(values, dtype=None):
        return np.array(values, dtype=dtype)

    @staticmethod
    def arange(start, end=None, dtype=None):
        if end is None:
            return np.arange(start, dtype=dtype)
        return np.arange(start, end, dtype=dtype)

    @staticmethod
    def meshgrid(*args, indexing="ij"):
        return np.meshgrid(*args, indexing=indexing)

    @staticmethod
    def logical_or(a, b):
        return np.logical_or(a, b)

    @staticmethod
    def logical_and(a, b):
        return np.logical_and(a, b)

    @staticmethod
    def any(x):
        return np.any(x)


torch = TorchCompat()
