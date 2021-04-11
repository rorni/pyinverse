# -*- coding: utf-8 -*-

import importlib
import warnings
import numpy as np


try:
    cp = importlib.import_module('cupy')
except ImportError:
    cp = None


def maximize_likelihood_EM(y, proj, x0=None, max_iter=100, reltol=1.e-8, 
                           smooth=None, cuda=False):
    """Finds MLE solution for Poisson distribution by EM algorithm.

    Parameters
    ----------
    y : array_like
        Observed data sample. It is an array of length M of floats or ints.
    proj : array_like
        Projection operator matrix. It has shape (M, N). 
    x0 : array_like
        Initial guess of parameters. It is an array of length N of floats.
        Default: None - vector of ones.
    max_iter : int
        Maximum number of iterations.
    reltol : float
        Relative tolerance for logarithmic likelihood function. When condition
        `|(logp[-1] - logp[-2]) / logp[-2]| < reltol` satisfied, optimization
        finished.
    smooth : array_like
        Smoothing operator. It has (N, N) shape and is applied to x. 
        Default: None - no smoothing.
    cuda : bool 
        Whether to use CUDA. Default: False.

    Returns
    -------
    x : numpy.ndarray
        The result. Array of size N.
    logp : numpy.ndarray
        Values of logarithmic likelihood function for each iteration.
    """
    y = np.array(y)
    proj = np.array(proj)

    M, N = proj.shape

    if x0 is None:
        x0 = np.ones(N)
    else:
        x0 = np.array(x0)

    if y.size != M:
        raise ValueError('Inconsistent shape of observed data: {0}.'.format(y.size))
    if x0.size != N:
        raise ValueError('Inconsistent shape of initial vector: {0}.'.format(x0.size))
    if smooth is None:
        smooth = np.eye(N)
    if smooth.shape != (N, N):
        raise ValueError('Inconsistent shape of smooth operator: {0}'.format(smooth.shape))

    index_row, index_col = get_nonzero_indices(proj)

    proj = proj[index_row, :][:, index_col]
    y = y[index_row]

    x1 = x0[index_col]

    if cuda and cp:
        x_new, logp = _mle_em_cuda(y, proj, x1, max_iter, reltol, smooth)
    else:
        if cuda:
            warnings.warn('CUDA is unavailable. CPU will be used.')
        x_new, logp = _mle_em(y, proj, x1, max_iter, reltol, smooth)

    x = x0.copy()
    x[index_col] = x_new

    return x, logp


def _mle_em(y, proj, x0, max_iter, reltol, smooth):
    S = 1.0 / np.sum(proj, axis=0)
    sum_log_k = np.sum([factorial_log(yi) for yi in y])
    x_old = x0
    iter = 0
    logp = []
    while True:
        denom = np.dot(proj, x_old)
        coeff = np.dot(y / denom, proj)
        x_new = x_old.copy()
        x_new = x_old * S * coeff

        x_new = np.dot(smooth, x_new)

        lam = np.dot(proj, x_new)
        logp.append(np.dot(y, np.log(lam)) - np.sum(lam) - sum_log_k)

        if iter == max_iter:
            break
        if iter >= 2 and np.abs((logp[-1] - logp[-2]) / logp[-2]) < reltol:
            break

        x_old = x_new
        iter += 1
    return x_new, np.array(logp)


def _mle_em_cuda(y, proj, x0, max_iter, reltol, smooth):
    sum_log_k = np.sum([factorial_log(yi) for yi in y])
    proj = cp.asarray(proj)
    y = cp.asarray(y)
    x_old = cp.asarray(x0)
    S = cp.divide(1, cp.sum(proj, axis=0))
    iter = 0
    logp = []
    while True:
        denom = cp.dot(proj, x_old)
        coeff = cp.dot(cp.divide(y, denom), proj)
        x_new = x_old.copy()
        x_new = x_old * S * coeff

        x_new = cp.dot(smooth, x_new)
        
        lam = cp.dot(proj, x_new)
        logp_cur = cp.dot(y, cp.log(lam)) - cp.sum(lam) - sum_log_k
        logp.append(logp_cur.get())

        if iter == max_iter:
            break
        if iter >= 2 and cp.abs((logp[-1] - logp[-2]) / logp[-2]) < reltol:
            break

        x_old = x_new
        iter += 1
    return cp.asnumpy(x_new), np.array(logp)


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

