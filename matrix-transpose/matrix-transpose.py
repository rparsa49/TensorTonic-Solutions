import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    A = np.asarray(A)
    rows = A.shape[0]
    cols = A.shape[1]

    res = [[0 for _ in range(rows)] for _ in range(cols)]
    res = np.asarray(res)

    for i in range(rows):
        for j in range(cols):
            res[j][i] = A[i][j]
            
    return res
