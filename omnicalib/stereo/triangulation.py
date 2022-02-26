import torch
from autograd import numpy as np, jacobian
from scipy.optimize import least_squares
from torch import Tensor

from ..geometry import dot
from .reprojection import project_points


def get_midpoint(vectors0: Tensor, vectors1: Tensor, transform: Tensor) -> Tensor:
    '''
    vectors0, vectors1 (n, 3)
    transform (3, 4) transform from coordinate system 0 to coordinate system 1
    returns midpoints (n, 3)
    see https://en.wikipedia.org/wiki/Skew_lines#Nearest_points
    '''
    R = transform[..., :3, :3]
    # inverse rotation
    vectors1 = vectors1 @ R
    n = torch.cross(vectors0, vectors1, dim=-1)
    n0 = torch.cross(vectors0, n, dim=-1)
    n1 = torch.cross(vectors1, n, dim=-1)
    # inverse rotation
    origin1 = R.t() @ -transform[..., :3, 3]
    c0 = dot(origin1, n1, keepdim=True) / \
        dot(vectors0, n1, keepdim=True) * vectors0
    c1 = origin1 + dot(-origin1, n0, keepdim=True) / \
        dot(vectors1, n0, keepdim=True) * vectors1
    return (c0 + c1) / 2


def optimize_midpoint(
    poly_theta0: Tensor,
    principal_point0: Tensor,
    vectors0: Tensor,
    image_points0: Tensor,
    poly_theta1: Tensor,
    principal_point1: Tensor,
    vectors1: Tensor,
    image_points1: Tensor,
    transform: Tensor):

    midpoints = get_midpoint(vectors0, vectors1, transform).cpu().numpy()

    poly_theta0 = poly_theta0.cpu().numpy()
    principal_point0 = principal_point0.cpu().numpy()
    image_points0 = image_points0.cpu().numpy()

    poly_theta1 = poly_theta1.cpu().numpy()
    principal_point1 = principal_point1.cpu().numpy()
    image_points1 = image_points1.cpu().numpy()

    transform = transform.cpu().numpy()

    def f(x):
        midpoints = x.reshape(-1, 3)
        projected_points0 = project_points(
            poly_theta0,
            principal_point0,
            np.eye(4),
            midpoints
        )
        projected_points1 = project_points(
            poly_theta1,
            principal_point1,
            transform,
            midpoints
        )
        residuals = np.stack((
            projected_points0 - image_points0,
            projected_points1 - image_points1
        )).flatten()
        # print mean distance in image coordinates
        # print(np.sqrt(np.square(residuals.reshape(-1, 2)).sum(axis=1)).mean())
        return residuals

    result = least_squares(
        f,
        midpoints.flatten(),
        jacobian(f),
        method='lm',
        verbose=2
    )
    return torch.from_numpy(result.x).view(vectors0.shape)
