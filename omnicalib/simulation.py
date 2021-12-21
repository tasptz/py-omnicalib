from typing import Tuple

import torch
from torch import Tensor

from .geometry import get_theta, unit


def project_ideal(focal_length: float, view_points: Tensor, model: str) \
        -> Tensor:
    '''
    Ideal projection
    '''
    n = unit(view_points)
    theta = get_theta(n, normed=True)
    if model == 'equidistant':
        r = focal_length * theta
    elif model == 'stereographic':
        r = 2 * focal_length * torch.tan(theta / 2)
    elif model == 'orthographic':
        r = focal_length * torch.sin(theta)
    elif model == 'equisolid':
        r = 2 * focal_length * torch.sin(theta / 2)
    else:
        raise KeyError()
    return unit(n[..., :2]) * r


def r_ideal(theta: Tensor, focal_length: float, model: str) -> Tensor:
    '''
    Ideal radius given incident angle `theta`
    '''
    if model == 'equidistant':
        r = theta * focal_length
    elif model == 'stereographic':
        r = torch.tan(theta / 2) * 2 * focal_length
    elif model == 'orthographic':
        r = torch.sin(theta) * focal_length
    elif model == 'equisolid':
        r = torch.sin(theta / 2) * 2 * focal_length
    else:
        raise KeyError()
    return r


def theta_ideal(r: Tensor, focal_length: float, model: str) -> Tensor:
    '''
    Ideal incident angle `theta` given radius `r`
    '''
    if model == 'equidistant':
        theta = r / focal_length
    elif model == 'stereographic':
        theta = 2 * torch.atan(r / (2 * focal_length))
    elif model == 'orthographic':
        theta = torch.asin(r / focal_length)
    elif model == 'equisolid':
        theta = 2 * torch.arcsin(r / (2 * focal_length))
    else:
        raise KeyError()
    return theta


def get_view_matrix(origin: Tensor, look_at: Tensor, down: Tensor) \
        -> Tuple[Tensor, Tensor]:
    '''
    Right handed coordinate system with
     * x facing right
     * y facing downwards
     * z in viewing direction
    '''
    view = look_at - origin
    view /= torch.linalg.norm(view, dim=-1, keepdim=True)
    down /= torch.linalg.norm(down, dim=-1, keepdim=True)
    B = len(origin)
    right = torch.cross(down, view, dim=-1)
    right /= torch.linalg.norm(right, dim=-1, keepdim=True)
    down = torch.cross(view, right, dim=-1)
    down /= torch.linalg.norm(down, dim=-1, keepdim=True)
    T = torch.eye(4)[None].repeat(B, 1, 1).to(origin)
    T[:, :3, 3] = -origin
    R = torch.eye(4)[None].repeat(B, 1, 1).to(origin)
    R[:, 0, :3] = right
    R[:, 1, :3] = down
    R[:, 2, :3] = view

    det = torch.linalg.det(R)
    assert torch.allclose(det, torch.ones_like(det))

    M = (R @ T)[:, :3]
    return M[:, :3, :3], M[:, :3, 3]
