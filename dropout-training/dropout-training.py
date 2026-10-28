import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    seed = 123,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    x = np.asarray(x)
    if rng is None:
        rng = np.random.default_rng(seed)

    # mask
    dropout_pattern = rng.random(x.shape)
    dropout_pattern = (dropout_pattern >= p).astype(float)
    # scale
    dropout_pattern = dropout_pattern * (1 / (1-p))
    
    output = x * dropout_pattern

    return np.asarray(output), np.asarray(dropout_pattern)

    
    