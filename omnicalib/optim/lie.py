'''
Lie algebra for 3d rigid transformations. See:
https://ethaneade.com/lie.pdf
https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf
'''

import math
from functools import cache
from typing import Tuple

from autograd import numpy as np
from autograd.extend import defvjp, primitive


def dot(a: np.ndarray, b: np.ndarray, keepdims=False) -> np.ndarray:
    '''
    Dot product over last axis
    '''
    return (a * b).sum(axis=-1, keepdims=keepdims)


@cache
def inv_factorial(x: int) -> int:
    '''
    Cached inverse factorial
    '''
    return 1 / math.factorial(x)


def hat(x: np.ndarray) -> np.ndarray:
    '''
    From vector in last axis to skew symmetric matrix
    '''
    x_hat = np.zeros_like(x, shape=x.shape + (3,))
    x_hat[..., 0, 1] = -x[..., 2]
    x_hat[..., 0, 2] = x[..., 1]

    x_hat[..., 1, 0] = x[..., 2]
    x_hat[..., 1, 2] = -x[..., 0]

    x_hat[..., 2, 0] = -x[..., 1]
    x_hat[..., 2, 1] = x[..., 0]
    return x_hat


def hat_inv(x_hat: np.ndarray) -> np.ndarray:
    '''
    From skew symmetric matrix in last two axis to vector
    '''
    x = np.zeros_like(x_hat, shape=x_hat.shape[:-1])
    x[..., 0] = x_hat[..., 2, 1]
    x[..., 1] = x_hat[..., 0, 2]
    x[..., 2] = x_hat[..., 1, 0]
    return x


def get_RV(o_hat: np.ndarray, phi2: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    '''
    `R` and `V` matrix from `hat(o)` and `phi**2`
    '''
    R = np.zeros_like(o_hat) + np.eye(3, dtype=o_hat.dtype)
    V = np.zeros_like(o_hat) + np.eye(3, dtype=o_hat.dtype)
    # mask for taylor series
    approx = phi2 < 1e-12
    exact = ~approx

    phi2_approx = phi2[approx][..., None, None]
    phi4 = phi2_approx**2
    o_hat_approx = o_hat[approx]
    o_hat_approx2 = o_hat_approx @ o_hat_approx
    A_approx = 1 - phi2_approx * inv_factorial(3) + phi4 * inv_factorial(5)
    B_approx = 0.5 - phi2_approx * inv_factorial(4) + phi4 * inv_factorial(6)
    C_approx = inv_factorial(3) - phi2_approx * \
        inv_factorial(5) + phi4 * inv_factorial(7)
    R[approx] += A_approx * o_hat_approx + B_approx * o_hat_approx2
    V[approx] += B_approx * o_hat_approx + C_approx * o_hat_approx2

    phi_exact = np.sqrt(phi2[exact])[..., None, None]
    phi_exact2 = phi2[exact][..., None, None]
    o_hat_exact = o_hat[exact]
    o_hat_exact2 = o_hat_exact @ o_hat_exact
    A = np.sin(phi_exact) / phi_exact
    B = (1 - np.cos(phi_exact)) / phi_exact2
    C = (1 - A) / phi_exact2
    R[exact] += A * o_hat_exact + B * o_hat_exact2
    V[exact] += B * o_hat_exact + C * o_hat_exact2
    return R, V


def exp(v: np.ndarray) -> np.ndarray:
    '''
    Exponential operator from algebra to manifold (`(R|t)` matrix)
    '''
    assert isinstance(v, np.ndarray)
    u = v[..., :3]
    o = v[..., 3:]

    o_hat = hat(o)
    phi2 = dot(o, o)
    R, V = get_RV(o_hat, phi2)
    t = V @ u[..., None]
    return np.concatenate((R, t), axis=-1)


@primitive
def _exp_null(v: np.ndarray) -> np.ndarray:
    '''
    Exponential operator from algebra to manifold (`(R|t)` matrix)
    for `v = 0`
    https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf
    Formula 10.10
    '''
    n = len(v) // 6
    return np.tile(np.eye(4, dtype=v.dtype)[None, :3], (n, 1, 1)).flatten()


def _exp_null_vjp(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    '''
    Performs the vector - jacobian product for the jacobian of
    the exponential operator `exp(x)` at `x = 0`
    https://j-towns.github.io/2017/06/12/A-new-trick.html
    `vjp` performs the v @ J operation
    '''
    e = np.eye(3, dtype=x.dtype)
    # jacobian ((4x3) x 6)
    J = np.zeros_like(x, shape=(12, 6))
    J[:9, -3:] = hat(-e).reshape(9, 3)
    J[9:, -6: -3] = e
    # jacobian to (6 x (3x4))
    J = J.reshape(4, 3, 6).swapaxes(0, 1).reshape(12, 6)

    def vjp(v: np.ndarray):
        '''
        vector `v` - jacobian `J` product
        '''
        return (v.reshape(-1, 1, 12) @ J).flatten()
    return vjp


defvjp(_exp_null, _exp_null_vjp)


def exp_null(v: np.ndarray) -> np.ndarray:
    '''
    Wrapper around exponential operator
    for reshaping
    '''
    assert np.allclose(v, 0.)
    M = _exp_null(v.flatten())
    return M.reshape(v.shape[:-1] + (3, 4))
