import torch
from autograd import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.polynomial import polyval
from torch import Tensor

from .geometry import get_theta, unit


def project_poly_rz(view_points: Tensor, poly: Tensor,
                    principal_point: Tensor = None) -> Tensor:
    '''
    Project `view_points` to image points with polynom
    that projects from radius to `z`
    `(x, y, z)` is the view vector
    radius is the length of `(x, y)` vector
    '''
    B, C = view_points.shape[:2]

    norm_xy = torch.linalg.norm(view_points[..., :2], dim=-1)
    z = view_points[..., -1] / norm_xy

    poly = poly[None, None].expand(B, C, -1).clone()
    poly[..., 1] -= z

    rho = []
    for p in poly.view(B * C, -1):
        full_roots = Polynomial(p.numpy(), domain=(-1, 1)).roots()
        roots = full_roots[np.logical_and(
            np.isreal(full_roots), full_roots > 0)].tolist()
        roots = [float(np.real(v)) for v in roots]
        if not roots:
            rho.append(0.)
        else:
            rho.append(min(roots))
    rho = torch.tensor(rho, dtype=torch.float64).view(B, C, 1)
    xy = (view_points[..., :2] / norm_xy[..., None]) * rho
    if principal_point is not None:
        xy = xy + principal_point
    return xy


def project_poly_thetar(view_points: Tensor, poly_theta: Tensor,
                        principal_point: Tensor = None,
                        normed: bool = False) -> Tensor:
    '''
    Project `view_points` to image points with polynom
    that projects from incident angle`theta` to radius
    `(x, y, z)` is the view vector with angle`theta` to `(0, 0, 1)`
    radius is the length of `(x, y)` vector
    '''
    theta = get_theta(view_points, normed)
    rho = polyval(theta, poly_theta)
    xy = unit(view_points[..., :2]) * rho
    if principal_point is not None:
        xy = xy + principal_point
    return xy
