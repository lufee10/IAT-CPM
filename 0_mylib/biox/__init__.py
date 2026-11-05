__version__ = "4.0.1"   # 这就是你当前这个文件夹的版本号

import numpy as np
from scipy import stats

from . import helpers
from .helpers import (
    _check_subject_alignment,
    _select_finite_variance_columns,
    _validate_array,
    _validate_boolean,
    _validate_finite_float,
    _validate_integer,
)

def pairedPearsonGp(X, Y, within=False):
# This function aims to measure the Pearson Similarity between each pair of subjects in two 
#   groups ("X" and "Y" groups), which yields a 2d-array as the similarity matrix.

# Note that if "X" and "Y" are the same group, the diagonal elements of similarity matrix would be "1", 
#   which could be eliminated by setting the parameter "within=True". In this case, the column number  
#   of the matrix would reduce by one.
 
    '''
    Parameters
    ----------
    X : 2d-array
        Each row contains the "edge vector" of one subject within group.
    Y : 2d-array
        Each row contains the "edge vector" of one subject within group.
    within : Bool
        Determine whether to eliminate the diagonal elements of similarity matrix. 

    Returns
    -------
    simiMatrix: 2d-array
        The similarity matrix between "X" and "Y" group, where each row and column refer to one subject 
        within "X" and "Y", respectively.
        
    '''

    simiMatrix = []
    m = X.shape[0]
    n = Y.shape[0]
    for i in range(m):
        tmpt = []
        for j in range(n):
            tmpt.append(stats.pearsonr(X[i], X[j])[0])
        simiMatrix.append(tmpt)

    if within: # To eliminate the diagonal elements of "simiMatrix".
        for i in range(m):
            del simiMatrix[i][i]

    return np.array(simiMatrix)



def edgeLoadingGp(fc_edges_gp):
# This function aims to measure the edge loading for each subject within group ("fc_edges_gp").
 
    '''
    Parameters
    ----------
    fc_edges_gp : 2d-array
        Each row contains the "edge vector" of one subject within group.


    Returns
    -------
    edgeLoadingGp: 2d-array
        Each row contains the "edgeLoading" of one subject within group.
        
    '''
    # To conduct the Z-normalization for each edge vector of subject within group.
    zGp = np.array([stats.zscore(e) for e in fc_edges_gp])  

    n = fc_edges_gp.shape[0]
    edgeLoadingGp = []
    for i in range(n):
        tmpt = []
        ind = list(range(n))
        del ind[i]
        for j in ind:
            tmpt.append(zGp[i] * zGp[j])
        edgeLoadingGp.append(np.mean(tmpt, axis=0))

    return np.array(edgeLoadingGp)
