from typing import Tuple

import torch
from torch import Tensor

from .geometry import check_origin, dot, gram_schmidt, null_space, unit
from .polyfit import polyval


def partial_extrinsics(image_points: Tensor, world_points: Tensor) \
        -> Tuple[Tensor, Tensor]:
    '''
    Solve for partial extrinsics
    `2 x 3` submatrix of a full rotation + translation matrix
    '''
    world_points = world_points[..., :2]
    std_world = torch.std(world_points, dim=(1, 2), keepdim=True)
    std_image = torch.std(image_points, dim=(1, 2), keepdim=True)
    image_points = image_points * std_world / std_image
    u, v = image_points.permute(2, 0, 1)
    x, y = world_points.permute(2, 0, 1)

    M = torch.stack((
        -v * x, -v * y, u * x, u * y, -v, u
    ), dim=2)
    nu = null_space(M)
    R = nu[:, :4].view(-1, 2, 2)
    T = nu[:, 4:]
    return R, T


def full_extrinsics(poly: Tensor, image_points: Tensor, word_points: Tensor) \
        -> Tuple[Tensor, Tensor]:
    '''
    Solve for full extrinisics with given polynom
    Check if solution is in front of chessboard marker plane,
    otherwise flip
    '''
    B, C = image_points.shape[:2]
    u, v = image_points.permute(2, 0, 1)
    x, y = word_points.permute(2, 0, 1)[:2]
    rho = polyval(torch.linalg.norm(image_points, dim=-1), poly)
    M = image_points.new_zeros((B, C, 3, 9))
    M[..., 0, 0] = rho * x
    M[..., 0, 1] = rho * y
    M[..., 0, 2] = rho
    M[..., 0, 6] = -u * x
    M[..., 0, 7] = -u * y
    M[..., 0, 8] = -u

    M[..., 1, 0] = -v * x
    M[..., 1, 1] = -v * y
    M[..., 1, 2] = -v
    M[..., 1, 3] = u * x
    M[..., 1, 4] = u * y
    M[..., 1, 5] = u

    M[..., 2, 3] = -rho * x
    M[..., 2, 4] = -rho * y
    M[..., 2, 5] = -rho
    M[..., 2, 6] = v * x
    M[..., 2, 7] = v * y
    M[..., 2, 8] = v
    M = M.view(B, C * 3, 9)

    def f(nu):
        R_par = nu[:, :, :2]
        R_norm = torch.linalg.norm(R_par, dim=1).mean(dim=-1)
        R_par = gram_schmidt(R_par)
        r0 = R_par[..., 0]
        r1 = R_par[..., 1]
        assert dot(r0, r1).abs().max() < 1e-3
        r2 = unit(torch.cross(r0, r1, dim=-1))
        R = torch.cat((R_par, r2[..., None]), dim=-1)
        assert torch.allclose(torch.linalg.det(
            R), R.new_ones(len(R)), atol=1e-2)
        T = nu[:, :, 2] / R_norm[..., None]
        return R, T

    nu = null_space(M)
    nu = nu.view(B, 3, 3)
    R, T = f(nu)
    # would origin lie behind chessboard (z > 0)
    mask_invert = ~check_origin(R, T)
    if mask_invert.any():
        R[mask_invert], T[mask_invert] = f(-nu[mask_invert])
    return R, T
