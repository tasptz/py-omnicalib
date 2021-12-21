from autograd import jacobian
from autograd import numpy as np
from numdifftools import Jacobian
from omnicalib.optim.lie import _exp_null, exp, hat, hat_inv


def test_hat():
    x = np.array((0.3, -0.1, 0.5))
    x_hat = hat(np.tile(x[None, None, None], (5, 7, 11, 1)))
    assert np.allclose(
        x_hat,
        np.array((
            (0, -0.5, -0.1),
            (0.5,    0, -0.3),
            (0.1,  0.3,    0)
        ), dtype=x.dtype)
    )


def test_hat_inv():
    x = np.random.uniform(-1, 1, (5, 7, 11, 3))
    assert np.allclose(x, hat_inv(hat(x)))


def test_exp():
    shape = (5, 7, 6)
    M = exp(np.zeros(shape, dtype=np.float64))
    R = M[..., :3, -4:-1]
    t = M[..., :3, -1]
    assert np.allclose(R, np.eye(3, dtype=M.dtype))
    assert np.allclose(t, np.zeros_like(t))


def test_exp_so3():
    R = np.array((
        (0.9924, 0.0078, -0.1228),
        (-0.0083, 1.0000, -0.0033),
        (0.1227, 0.0043,  0.9924)
    ))
    v = np.array((
        0,
        0,
        0,
        3.76358959e-03,
        -1.23075928e-01,
        -8.09421618e-03
    ))
    R_test = exp(v[None])[0, :3, :3]
    assert np.allclose(R_test, R, atol=1e-3)


def test_exp_grad():
    v = np.zeros((1, 6), dtype=np.float64)

    def f(x):
        return exp(x.reshape(v.shape)).flatten()
    g_num = Jacobian(f)(v.flatten())
    g = jacobian(_exp_null)(v.flatten())
    assert np.allclose(g_num, g)
