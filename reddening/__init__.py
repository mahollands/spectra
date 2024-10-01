from .CCM89 import A_CCM89
from .FM90 import A_FM90
from .G23 import A_G23

__all__ = [
    "A_curve"
]

extinction_models = {
    'CCM89': A_CCM89,
    'FM90': A_FM90,
    'G23': A_G23,
}


def A_curve(lam, R=3.1, use_model=None):
    model = extinction_models[use_model]
    return model(lam, R)
