import numpy as np


def quadratic(x, m=0.5, a=0.1):
    return a * (x - m) ** 2


def absolute_value(x, m, a):
    return a * np.abs(x - m)


def linear(x, m, a):
    return a * (x - m)
