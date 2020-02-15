# coding: utf-8
import numpy as np


def AND(x1, x2):
    x, w, b = np.array([x1, x2]), np.array([0.5, 0.5]), -0.7
    return 1 if np.sum(x * w) + b > 0 else 0


if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = AND(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))
