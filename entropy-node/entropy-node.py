import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return float(-np.sum(p*np.log2(p)))