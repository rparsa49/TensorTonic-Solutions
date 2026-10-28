import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    w = np.asarray(w)
    g = np.asarray(g)
    s = np.asarray(s)

    # update running avg
    s = beta * s + (1-beta) * (g**2)
    # update parameter
    w = w - ((lr / (s + eps) ** 0.5)) * g
    return w, s
    