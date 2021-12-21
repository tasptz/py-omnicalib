from typing import Tuple, Union

import torch
from autograd import jacobian
from autograd import numpy as np
from autograd.numpy import ndarray
from autograd.numpy.numpy_boxes import ArrayBox
from scipy.optimize import least_squares
from torch import Tensor
from tqdm import tqdm

from ..polyfit import poly_r_to_theta
from .lie import exp, exp_null, dot


def _unit(x: np.ndarray) -> np.ndarray:
    '''
    Make unit vectors in last axis
    '''
    return x / np.linalg.norm(x, axis=-1)[..., None]


def _hom_mat(x: np.ndarray) -> np.ndarray:
    '''
    Extend matrices in last 2 axis for homogeneous
    vectors
    '''
    hom = np.tile(
        np.array((0, 0, 0, 1), dtype=x.dtype),
        x.shape[:-2] + (1, 1)
    )
    return np.concatenate((x, hom), axis=-2)


def _hom(x: np.ndarray) -> np.ndarray:
    '''
    Make vectors in last axis homogeneous
    '''
    return np.concatenate(
        (x, np.ones(x.shape[:-1] + (1,), dtype=x.dtype)),
        axis=-1
    )


def _get_theta(p: np.ndarray, normed: bool = False) -> np.ndarray:
    '''
    Incident angle:
    Angle between `p` and `(0, 0, 1)`
    '''
    if not normed:
        p = _unit(p)
    return np.arccos(dot(p, np.array((0, 0, 1), dtype=p.dtype), True))


def reprojection(R: Tensor, T: Tensor, poly: Tensor,
                 principal_point_initial: Tensor,
                 image_points: Tensor, world_points: Tensor) \
        -> Tuple[bool, Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:
    '''
    Optimize the reprojection error via Levenberg–Marquardt algorithm
    '''
    extrinsics = _hom_mat(torch.cat((R, T[..., None]), dim=-1).cpu().numpy())
    image_points_np = image_points.cpu().numpy()
    world_points_np = world_points.cpu().numpy()

    def residuals(v: Union[np.ndarray, ArrayBox], poly_theta: np.ndarray,
                  principal_point: np.ndarray) -> np.ndarray:
        nonlocal extrinsics
        if isinstance(v, np.ndarray):
            E = exp(v) @ extrinsics
        else:
            extrinsics = _hom_mat(exp(v._value) @ extrinsics)
            v._value.fill(0.)
            E = exp_null(v) @ extrinsics
        view_points = _hom(world_points_np) @ E.swapaxes(-1, -2)
        theta = _get_theta(view_points)

        # evaluate poly (poly has no absolute component)
        r = dot(
            np.stack([theta] +
                     [theta**i for i in range(2, len(poly_theta) + 1)],
                     axis=len(theta.shape)),
            poly_theta
        )
        projected_points = _unit(view_points[..., :2]) * r + principal_point
        return (projected_points - image_points_np).flatten()

    def unpack(x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Unpack flattened parameters to variables
        '''
        B = len(R)
        v = x0[:B * 6].reshape(B, 6)
        poly_theta = x0[B * 6: -2]
        principal_point = x0[-2:]
        return v, poly_theta, principal_point

    poly_theta = poly_r_to_theta(poly, image_points, principal_point_initial)
    principal_point = principal_point_initial.cpu().numpy()

    x0 = np.concatenate([R.new_zeros(len(R) * 6).numpy(),
                        poly_theta[1:], principal_point])

    with tqdm(ascii=True) as progress:  # default number of evaluations
        def fun(x: np.ndarray) -> np.ndarray:
            '''
            Calculate residuals from flattened parameters
            '''
            r = residuals(*unpack(x))
            r_rmse = np.sqrt(
                ((r if isinstance(r, ndarray) else r._value)**2).mean())
            progress.set_description(f'rmse {r_rmse:.3e}')
            progress.update()
            return r
        result = least_squares(
            fun,
            x0=x0,
            jac=jacobian(fun),
            method='lm',
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10
        )
    reprojection_errors = torch.from_numpy(np.sqrt(
        (residuals(*unpack(result.x)).reshape(-1, 2)**2).sum(axis=1)))

    v, poly_theta, principal_point = unpack(result.x)
    M = exp(v) @ extrinsics
    M, poly_theta, principal_point = [torch.from_numpy(
        x).to(R) for x in (M, poly_theta, principal_point)]
    poly_theta = torch.cat((poly_theta.new_zeros(1), poly_theta))
    R = M[..., -4:-1]
    T = M[..., -1]
    return result.success, reprojection_errors, \
        (R, T, poly_theta, principal_point)
