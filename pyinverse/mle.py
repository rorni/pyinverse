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
    if proj.shape != (N, M):
        raise ValueError('Inconsistent shape of proj: {0}.'.format(proj.shape))
    if not smooth:
        smooth = np.eye(N)
    if smooth.shape != (N, N):
        raise ValueError('Inconsistent shape of smooth operator: {0}'.format(smooth.shape))

    S = 1.0 / np.sum(proj, axis=0)

    x_old = x0
    iter = 0

    while True:
        denom = np.dot(proj, x_old)
        coeff = np.dot(y / denom, proj)
        x_new = x_old * S * coeff
        
        x_new = np.dot(smooth, x_new)

        if iter == max_iter:
            break
        if np.linalg.norm(y, np.dot(proj, x_new)) < abstol:
            break

        x_old = x_new
        iter += 1
    return x_new

