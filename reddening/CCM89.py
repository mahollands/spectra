import numpy as np

C_a = [1, +0.17699, -0.50447, -0.02427, +0.72085, +0.01979, -0.77530, +0.32999]
C_b = [0, +1.41338, +2.28305, +1.07233, -5.38434, -0.62251, +5.30260, -2.09002]
poly_a = np.polynomial.Polynomial(C_a)
poly_b = np.polynomial.Polynomial(C_b)

poly_Fa = np.polynomial.Polynomial([-0.04473, -0.009779])
poly_Fb = np.polynomial.Polynomial([+0.21300, +0.120700])


def A_CCM89(lam, R):
    """
    Calculate CCM 1989 extinction curve. x is in units of 1/um.
    """
    x = 1/lam

    FIR = (x < 0.3)
    IR  = (x >= 0.3) & (x < 1.1)
    opt = (x >= 1.1) & (x < 3.3)
    UV  = (x >= 3.3) & (x < 8.0)
    FUV = (x >= 8.0)

    a = np.zeros_like(x)
    b = np.zeros_like(x)
    a[FIR], b[FIR] = ab_IR(0.3)
    a[IR], b[IR] = ab_IR(x[IR])
    a[opt], b[opt] = ab_opt(x[opt])
    a[UV], b[UV] = ab_UV(x[UV])
    a[FUV], b[FUV] = ab_UV(8.0)
    A = a + b/R
    return A


def ab_IR(x):
    """
    0.3 <= x*um < 1.1
    """
    xp = x**1.61
    return 0.574*xp, -0.527*xp


def ab_opt(x):
    """
    1.1 <= x*um < 3.3
    """
    x_ = x-1.82
    return poly_a(x_), poly_b(x_)


def ab_UV(x):
    """
    3.3 <= x*um < 8.0
    """
    x_ = x-5.9
    Fa = x_**2 * np.where(x_ > 0, poly_Fa(x_), 0)
    Fb = x_**2 * np.where(x_ > 0, poly_Fb(x_), 0)
    a =  1.752 - 0.316*x - 0.104/((x-4.67)**2 + 0.341) + Fa
    b = -3.090 + 1.825*x + 1.206/((x-4.62)**2 + 0.263) + Fb
    return a, b
