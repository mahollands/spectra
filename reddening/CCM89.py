import numpy as np

poly_a = np.polynomial.Polynomial([1, +0.17699, -0.50447, -0.02427, +0.72085, \
    +0.01979, -0.77530, +0.32999])
poly_b = np.polynomial.Polynomial([0, +1.41338, +2.28305, +1.07233, -5.38434, \
    -0.62251, +5.30260, -2.09002])

poly_Fa = np.polynomial.Polynomial([0, 0, -0.04473, -0.009779])
poly_Fb = np.polynomial.Polynomial([0, 0, +0.21300, +0.120700])

def A_CCM89(x, R):
    """
    Calculate CCM 1989 extinction curve. x is in units of 1/um.
    """
    FIR = (x < 0.3)
    IR  = (x >= 0.3) & (x < 1.1)
    opt = (x >= 1.1) & (x < 3.3)
    UV  = (x >= 3.3) & (x < 8.0)
    FUV = (x >= 8.0)

    a = np.zeros_like(x)
    b = np.zeros_like(x)
    a[IR], b[IR] = Av_IR(x[IR])
    a[opt], b[opt] = Av_opt(x[opt])
    a[UV], b[UV] = Av_UV(x[UV])
    a[FIR], b[FIR] = Av_IR(0.3)
    a[FUV], b[FUV] = Av_UV(8.0)
    A = a + b/R
    return A

def Av_IR(x):
    """
    0.3 <= x/um < 1.1
    """
    return [c*x**1.161 for c in (0.574, -0.527)]

def Av_opt(x):
    """
    1.1 <= x/um < 3.3
    """
    return poly_a(x-1.82), poly_b(x-1.82)

def Av_UV(x):
    """
    3.3 <= x/um < 8.0
    """
    Fa, Fb = poly_Fa(x-5.9), poly_Fb(x-5.9)
    if isinstance(x, np.ndarray):
        Fa[x < 5.9], Fb[x < 5.9] = 0, 0
    elif isinstance(x, (int, float)):
        if x < 5.9:
            Fa = Fb = 0
    else:
        raise TypeError
    a =  1.752 - 0.316*x - 0.104/((x-4.67)**2 + 0.341) + Fa
    b = -3.090 + 1.825*x + 1.206/((x-4.62)**2 + 0.263) + Fb
    return a, b

