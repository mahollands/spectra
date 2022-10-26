from .CCM89 import A_CCM89

__all__ = [
    "A_curve"
]

def A_curve(x, R=3.1, use_model='CCM89'):
    extinction_models = {
        'CCM89' : A_CCM89,
    }
    model = extinction_models[use_model]
    return model(x, R)
