import numpy as np


def A_FM90(x, R):
    c2 = -0.824 + 4.717 / R
    c1 = 2.030 - 3.007 * c2
    c3, c4 = 3.23, 0.41
    return c1 + c2*x + c3*D(x, 4.596, 0.99) + c4*F(x)


def D(x, x0, gamma):
    return x**2/((x*gamma)**2 + (x**2-x0**2)**2)


def F(x):
    x_ = x-5.9
    f = 0.5392*x_**2 + 0.05644*x_**3
    return np.where(x < 5.9, 0, f)
