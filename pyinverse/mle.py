# -*- coding: utf-8 -*-

import numpy as np


def maximize_likelihood_EM(y, x0, proj, max_iter=100, abstol=1.e-8, smooth=None):
    """Finds MLE solution for Poisson distribution by EM algorithm.

    Parameters
    ----------
    y : array_like
        Observed data sample. It is an array of length M of floats or ints.
    x0 : array_like
        Initial guess of parameters. It is an array of length N of floats.
    proj : array_like
        Projection operator matrix. It has shape (M, N). 
    max_iter : int
        Maximum number of iterations.
    abstol : float or array_like
        Absolute tolerance. 
    smooth : array_like
        Smoothing operator. It has (N, N) shape and is applied to x. 
        Default: None - no smoothing.

    Returns
    -------
    x : numpy.ndarray
        The result. Array of size N.
    """
    y = np.array(y)
    x0 = np.array(x0)
    proj = np.array(proj)

    M = y.size
    N = x0.size
    if proj.shape != (M, N):
        raise ValueError('Inconsistent shape of proj: {0}.'.format(proj.shape))
    if smooth is None:
        smooth = np.eye(N)
    if smooth.shape != (N, N):
        raise ValueError('Inconsistent shape of smooth operator: {0}'.format(smooth.shape))

    index_row, index_col = get_nonzero_indices(proj)

    proj = proj[index_row, :][:, index_col]
    y = y[index_row]

    S = 1.0 / np.sum(proj, axis=0)

    sum_log_k = np.sum([factorial_log(yi) for yi in y])

    x_old = x0
    iter = 0

    while True:
        denom = np.dot(proj, x_old[index_col])
        coeff = np.dot(y / denom, proj)
        x_new = x_old.copy()
        x_new[index_col] = x_old[index_col] * S * coeff
        
        x_new = np.dot(smooth, x_new)

        lam = np.dot(proj, x_new[index_col])
        logp = np.dot(y, np.log(lam)) - np.sum(lam) - sum_log_k

        if iter == max_iter:
            break
        if np.linalg.norm(y - np.dot(proj, x_new[index_col])) < abstol:
            break

        x_old = x_new
        iter += 1
    return x_new, logp


def factorial_log(k):
    """Calculates log(k!)."""
    result = 0
    for i in range(1, k+1):
        result += np.log(i)
    return result


def get_nonzero_indices(proj):
    """Finds nonzero rows and columns of projection matrix.

    Parameters
    ----------
    proj : array_like
        Projection matrix.
    
    Returns
    -------
    index_row, index_column : array[bool]
        Row and column index arrays.
    """
    index_row = np.array(np.count_nonzero(proj, axis=1), dtype=bool)
    index_col = np.array(np.count_nonzero(proj, axis=0), dtype=bool)
    return index_row, index_col

