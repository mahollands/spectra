"""
Implementation of Gordon et al. 2023 extinciton model. This covers
912AA--32µm, and covers R(V) = 2.3--5.6.
https://ui.adsabs.harvard.edu/abs/2023ApJ...950...86G/abstract
"""
import numpy as np

#UV consts.
F_UV = np.polynomial.Polynomial([0, 0, 0.5392, 0.05644])
#opt consts
Ea = np.polynomial.Polynomial([-0.35848, 0.7122, 0.08746, -0.05403, 0.00674])
Eb = np.polynomial.Polynomial([0.12354, -2.68335, 2.01901, -0.39299, 0.03355])
opt_x0, opt_gamma = [2.288, 2.054, 1.587], [0.243, 0.179, 0.243]
Fa_opt, Fb_opt = [0.03893, 0.02965, 0.01747], [0.18453, 0.19728, 0.17130]

def Drude(x, x0, gamma):
    """
    Drude profile for adding bumps to extinction curve
    """
    p, q = x*x - x0*x0, gamma*x
    return x*x / (p*p + q*q)

def Drude_m(lam, lam0, gamma0, a):
    """
    Modified drude profile for adding asymmetric bumps to extinction curve
    """
    gamma = 2*gamma0/(1+np.exp(a*(lam-lam0)))
    numerator = (gamma/lam0)**2
    denominator = (lam/lam0 - lam0/lam)**2 + (gamma/lam0)**2
    return numerator/denominator

def W(lam, lam_b, delta):
    z = (lam - lam_b + 0.5*delta)/delta
    return np.select([z<0, z>1], [0, 1], z*z*(3-2*z))

def A_G23(lam, Rv, return_ab=False):
    """
    Calculate Gordon et al. 2023 extinction curve. lam is in units of um.
    """
    x = 1/lam
    sections = [
        (lam >= 0.0912) & (lam < 0.3), #UV
        (lam >= 0.3) & (lam < 0.33), #UV/opt
        (lam >= 0.33) & (lam < 0.9), #opt
        (lam >= 0.9) & (lam < 1.1), #opt/IR
        (lam >= 1.1) & (lam <= 32.0), #IR
    ]

    a_UV, b_UV = ab_UV(x)
    a_opt, b_opt = ab_opt(x)
    a_IR, b_IR = ab_IR(lam)

    W1, W2 = W(lam, 0.315, 0.03), W(lam, 1.0, 0.2)

    a = np.select(
        sections,
        [a_UV, (1-W1)*a_UV + W1*a_opt, a_opt, (1-W2)*a_opt + W2*a_IR, a_IR],
        0,
    )
    b = np.select(
        sections,
        [b_UV, (1-W1)*b_UV + W1*b_opt, b_opt, (1-W2)*b_opt + W2*b_IR, b_IR],
        0,
    )
    A = a + b*(1/Rv - 1/3.1)

    if return_ab:
        return A, a, b
    return A

def ab_UV(x):
    """
    0.0912 <= x/um < 0.30
    """
    F = np.where(x >= 5.9, F_UV(x-5.9), 0)
    D = Drude(x, 4.60, 0.99)
    a =  0.81297 + 0.27750*x + 1.06295*D + 0.11303*F
    b = -2.97868 + 1.89808*x + 3.10334*D + 0.65484*F
    return a, b

def ab_opt(x):
    """
    0.33 <= x/um < 1.1
    """
    Ds = [Drude(x, x0, gamma)*gamma**2 for x0, gamma in zip(opt_x0, opt_gamma)]
    a = Ea(x) + sum(F*D for F, D in zip(Fa_opt, Ds))
    b = Eb(x) + sum(F*D for F, D in zip(Fb_opt, Ds))
    return a, b

def ab_IR(lam):
    """
    0.9 <= x/um < 32
    """
    #note probable mistake on sign of α1b in paper
    α1a, α1b, α2 = 1.68467, 1.06099, 0.78791
    lam_b, delta = 4.30578, 4.78338
    W_ = W(lam, lam_b, delta)
    a =  0.38526 * (lam**-α1a * (1 - W_) + lam_b**(α2-α1a)*lam**-α2 * W_) \
        + 0.06652*Drude_m(lam,  9.843400,  2.21205, -0.24703) \
        + 0.02670*Drude_m(lam, 19.258294, 17.00000, -0.27000)
    b =  -1.01251 * lam**-α1b
    return a, b

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    #UV (Fig 4 of G23)
    x = np.geomspace(0.09, 0.33, 1000)
    a, b = ab_UV(1/x)
    plt.subplot(2, 1, 1)
    plt.plot(x, a)
    plt.semilogx()
    plt.xlim(0.09, 0.33)
    plt.ylim(1, 8)
    plt.subplot(2, 1, 2)
    plt.plot(x, b)
    plt.semilogx()
    plt.xlim(0.09, 0.33)
    plt.ylim(0, 50)
    plt.show()

    #opt (Fig 5 of G23)
    x = np.geomspace(0.3, 1.1, 1000)
    a, b = ab_opt(1/x)
    plt.subplot(2, 1, 1)
    plt.plot(x, a)
    plt.semilogx()
    plt.xlim(0.3, 1.1)
    plt.ylim(0.3, 2.0)
    plt.subplot(2, 1, 2)
    plt.plot(x, b)
    plt.semilogx()
    plt.xlim(0.3, 1.1)
    plt.ylim(-1.5, 6.0)
    plt.show()

    #IR (Fig 6 of G23)
    x = np.geomspace(1, 30, 1000)
    a, b = ab_IR(x)
    plt.subplot(2, 1, 1)
    plt.plot(x, a)
    plt.loglog()
    plt.xlim(1, 30)
    plt.ylim(0.01, 1)
    plt.subplot(2, 1, 2)
    plt.plot(x, b)
    plt.semilogx()
    plt.xlim(1, 30)
    plt.ylim(-1.5, 0.5)
    plt.show()

    #Attenuation
    x = np.geomspace(0.0912, 32, 1000)
    A, a, b = A_G23(x, 3.1, return_ab=True)
    plt.plot(x, A)
    plt.loglog()
    plt.show()
 
    #Variable Rv (Fig 8 of G23)
    for Rv in (2.5, 3.1, 4.0, 5.5):
        A, a, b = A_G23(x, Rv, return_ab=True)
        plt.plot(x, A)
    plt.loglog()
    plt.show()

    x1 = np.geomspace(0.0912, 0.35, 1000)
    a1, b1 = ab_UV(1/x1)
    x2 = np.geomspace(0.3, 1.1, 1000)
    a2, b2 = ab_opt(1/x2)
    x3 = np.geomspace(1, 32, 1000)
    a3, b3 = ab_IR(x3)

    #ab mergered
    plt.subplot(2, 1, 1)
    plt.plot(x1, a1)
    plt.plot(x2, a2)
    plt.plot(x3, a3)
    plt.plot(x, a)
    plt.loglog()
    plt.xlim(0.08, 35)
    plt.ylim(0.01, 20.0)
    plt.subplot(2, 1, 2)
    plt.plot(x1, b1)
    plt.plot(x2, b2)
    plt.plot(x3, b3)
    plt.plot(x, b)
    plt.semilogx()
    plt.xlim(0.08, 35)
    plt.ylim(-2, 20)
    plt.show()
