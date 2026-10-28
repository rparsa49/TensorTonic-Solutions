import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    param = np.asarray(param, dtype=float)
    grad = np.asarray(grad, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)

    # update first moment
    m = beta1 * m + (1-beta1) * grad

    # update second moment
    v = beta2 * v + (1-beta2) * (grad**2)

    # bias correction
    m_new = m / (1-(beta1**t))
    v_new = v / (1-(beta2**t))

    # parameter update
    param = param - lr * (m_new / ((v_new ** 0.5) + eps))

    return param, m, v