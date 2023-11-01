from .CCM89 import A_CCM89
from .FM90 import A_FM90

__all__ = [
    "A_curve"
]

extinction_models = {
    'CCM89' : A_CCM89,
    'FM90' : A_FM90
}

def A_curve(x, R=3.1, use_model='CCM89'):
    model = extinction_models[use_model]
    return model(x, R)
