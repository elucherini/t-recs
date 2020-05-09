import numpy as np
import pandas as pd


'''
'' Normalize matrix
'' @matrix: matrix to normalize
'' @axis: 1 -> rows, 0 -> columns
'''
def normalize_matrix(matrix, axis=1):
    divisor = matrix.sum(axis=axis)[:, None]
    result = np.divide(matrix, divisor, out=np.zeros(matrix.shape, dtype=float), where=divisor!=0)
    return result

def toDataFrame(data, index=None):
    if index is None:
        df = pd.DataFrame(data)
    else:
        df = pd.DataFrame(data).set_index(index)
    return df
