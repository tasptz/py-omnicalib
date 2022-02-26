from typing import Tuple, Union

import torch
from autograd import jacobian
from autograd import numpy as np
from autograd.numpy.numpy_boxes import ArrayBox
from scipy.optimize import least_squares
from torch import Tensor
from tqdm import tqdm

from ..optim.lie import dot, exp, exp_null
from ..optim.reprojection import _get_theta, _hom, _hom_mat, _unit


def project_points(poly_theta: np.ndarray, principal_point: np.ndarray,
                   M: Union[np.ndarray, ArrayBox], points: np.ndarray) \
        -> Union[np.ndarray, ArrayBox]:
    view_points = _hom(points) @ M[..., :3, :].swapaxes(-1, -2)
    theta = _get_theta(view_points)

    # evaluate poly (poly has no absolute component)
    theta_terms = np.concatenate([theta] +
                                 [theta **
                                  i for i in range(2, poly_theta.shape[-1])],
                                 axis=-1
                                 )
    r = dot(theta_terms, poly_theta[..., 1:], keepdims=True)
    projected_points = _unit(
        view_points[..., :2]) * r + principal_point
    return projected_points


def apply_offset_transform(v: Union[np.ndarray, ArrayBox], M: np.ndarray) \
        -> Union[np.ndarray, ArrayBox]:
    if isinstance(v, np.ndarray):
        return exp(v) @ M
    else:
        # overwrite M with updated version!
        np.copyto(_hom_mat(exp(v._value) @ M), M)
        v._value.fill(0.)
        return exp_null(v) @ M


def reprojection(extrinsics: Tensor, poly_thetas: Tensor,
                 principal_points: Tensor,
                 image_points: Tensor,
                 world_points: Tensor) \
        -> Tuple[Tensor, Tensor]:
    '''
    Optimize the reprojection error via Levenberg–Marquardt algorithm
    extrinsics (num_cameras, num_images, 4, 4)
    '''
    num_cam, num_img = extrinsics.shape[:2]

    E0 = extrinsics[0]
    relative_extrinsics = extrinsics[1:, 0] @ torch.linalg.inv(E0[:1])

    E0 = E0.cpu().numpy()
    relative_extrinsics = relative_extrinsics.cpu().numpy()

    poly_thetas = poly_thetas.cpu().numpy()
    principal_points = principal_points.cpu().numpy()
    image_points = image_points.cpu().numpy()
    world_points = world_points.cpu().numpy()

    def residuals(
            V0: Union[np.ndarray, ArrayBox],
            V_relative: Union[np.ndarray, ArrayBox]) -> np.ndarray:
        O0 = apply_offset_transform(V0, E0)
        O_relative = apply_offset_transform(V_relative, relative_extrinsics)

        M = np.stack((
            O0,
            O_relative @ _hom_mat(O0)
        ))

        projected_points = project_points(
            poly_thetas, principal_points[:, None, None], M, world_points)

        return (projected_points - image_points).flatten()

    def unpack(x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Unpack flattened parameters to variables
        '''
        V0 = x0[:num_img * 6].reshape(num_img, 6)
        V_relative = x0[num_img * 6:].reshape(num_cam - 1, 6)
        return V0, V_relative

    x0 = np.zeros((num_img + num_cam - 1) * 6, dtype=np.float64)
    with tqdm(ascii=True) as progress:  # default number of evaluations
        def fun(x: np.ndarray) -> np.ndarray:
            '''
            Calculate residuals from flattened parameters
            '''
            r = residuals(*unpack(x))
            r_rmse = np.sqrt(
                ((r if isinstance(r, np.ndarray) else r._value)**2).mean())
            progress.set_description(f'rmse {r_rmse:.3e}')
            progress.update()
            return r
        least_squares(
            fun,
            x0=x0,
            jac=jacobian(fun),
            method='lm',
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            verbose=2
        )
    return torch.from_numpy(E0), torch.from_numpy(relative_extrinsics)
