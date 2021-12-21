from typing import Generator, Tuple

import torch
from torch import Tensor


def hom(x: Tensor) -> Tensor:
    '''
    Homogeneous vectors in last axis
    '''
    return torch.cat((x, x.new_ones(size=x.shape[:-1] + (1,))), dim=-1)


def null_space(A: Tensor) -> Tensor:
    '''
    Solve for x so that A @ x = null
    '''
    U, S, Vh = torch.linalg.svd(A)
    # get the last row of Vh
    # (the last column of V)
    return Vh[..., -1, :]


def dot(a: Tensor, b: Tensor, keepdim=False) -> Tensor:
    '''
    Dot product in last axis
    '''
    return (a * b).sum(dim=-1, keepdim=keepdim)


def unit(v: Tensor, dim: int = -1) -> Tensor:
    '''
    Unit vectors
    '''
    return v / torch.linalg.norm(v, dim=dim, keepdims=True)


def transform(x: Tensor, R: Tensor, T: Tensor = None) -> Tensor:
    '''
    Transform points x with rotation R and translation T
    '''
    y = x @ R.transpose(-1, -2)
    return y if T is None else y + T[:, None]


def check_origin(R: Tensor, T: Tensor) -> Tensor:
    '''
    Check if origin has negative z coordinate
    after `(R|T)^-1`
    If so, origin is in front of `xy` plane defined
    by `(R|T)`
    '''
    return (-T[..., None, :] @ R)[..., 0, 2] < 0


def proj(a: Tensor, b: Tensor) -> Tensor:
    '''
    Project vectors `b` onto vectors a (last axis)
    (see Gram-Schmidt process)
    '''
    return dot(a, b, True) / dot(a, a, True) * a


def gram_schmidt(R: Tensor) -> Tensor:
    '''
    Gram-Schmidt process for two columns of rotation matrices
    (last 2 axis)
    '''
    r1 = unit(R[..., 0])
    r2 = unit(R[..., 1] - proj(R[..., 0], R[..., 1]))
    return torch.stack((r1, r2), dim=-1)


def get_theta(p: Tensor, normed: bool = False) -> Tensor:
    '''
    Angle between points `p` and `(0, 0, 1)`
    '''
    if not normed:
        p = unit(p)
    return torch.arccos(dot(p, p.new_tensor((0, 0, 1)), True))


def spiral(step: int = 1, end: int = 10) \
        -> Generator[Tuple[int, int], None, None]:
    '''
    Generates `(x, y)` coordinates by spiralling
    outwards from `(0, 0)` with step size `step`
    while `x < end` and `y < end`
    '''
    x = y = 0
    ma = 0
    stop = False
    while True:
        ma += step
        if ma >= end:
            ma = end - 1
            stop = True
        for xi in range(x, ma + 1, step):
            yield xi, y
        if stop:
            break
        x = xi
        for yi in range(y, ma + 1, step):
            yield x, yi
        y = yi
        for xi in range(x, -ma - 1, -step):
            yield xi, y
        x = xi
        for yi in range(y, -ma - 1, -step):
            yield x, yi
        y = yi
