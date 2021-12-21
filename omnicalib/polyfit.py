from typing import Tuple
import torch
from torch import Tensor

from .geometry import dot, get_theta


def polynom(x: Tensor, degree: int) -> Tensor:
    '''
    Raise `x` with polynomial degrees
    '''
    return torch.stack([
        torch.ones_like(x), x] + [torch.pow(x, i)
                                  for i in range(2, degree + 1)
                                  ], dim=-1)


def polyval(x: Tensor, poly: Tensor) -> Tensor:
    '''
    Evaluate polynom `poly` at `x`
    '''
    return dot(polynom(x, len(poly) - 1), poly)


def polyfit(degree: int, image_points: Tensor, world_points: Tensor,
            R: Tensor, T: Tensor) -> Tuple[Tensor, Tensor]:
    '''
    Fit `world_points` to `image_points` given extrinsics `R` and
    incomplete `T` using a polynom with given `degree`
    Note: the `z` component of `T` is determined in this step
    '''
    u, v = image_points.permute(2, 0, 1)
    x, y = world_points[..., :2].permute(2, 0, 1)

    r11 = R[:, 0, 0:1]
    r12 = R[:, 0, 1:2]

    r21 = R[:, 1, 0:1]
    r22 = R[:, 1, 1:2]

    r31 = R[:, 2, 0:1]
    r32 = R[:, 2, 1:2]

    t1 = T[:, 0:1]
    t2 = T[:, 1:2]

    a = r21 * x + r22 * y + t2
    b = v * (r31 * x + r32 * y)
    c = r11 * x + r12 * y + t1
    d = u * (r31 * x + r32 * y)

    rho = torch.sqrt(torch.pow(u, 2) + torch.pow(v, 2))
    rho_poly = polynom(rho, degree)
    rho_poly = torch.cat((rho_poly[..., 0:1], rho_poly[..., 2:]), dim=-1)
    N = rho_poly.shape[-1]

    B, C = image_points.shape[:2]
    M = image_points.new_zeros((B, 2 * C, N + B))

    M[:, :C, :N] = rho_poly * a[:, :, None].expand(rho_poly.shape)
    M[:, C:, :N] = rho_poly * c[:, :, None].expand(rho_poly.shape)
    for i in range(B):
        M[i, :C, N + i] = -v[i]
        M[i, C:, N + i] = -u[i]
    M = M.view(B * 2 * C, N + B)

    q = torch.cat((b, d), dim=1).view(B * 2 * C)
    solution, residuals, rank, singular_value = torch.linalg.lstsq(M, q)

    poly = image_points.new_zeros(degree + 1)
    poly[0] = solution[0]
    poly[2:] = solution[1:N]

    T = torch.cat((T, solution[N:, None]), dim=1)

    return poly, T


def poly_r_to_theta(poly: Tensor, image_points: Tensor,
                    principal_point: Tensor) -> Tensor:
    '''
    Convert polynom that projects radius to `z` to polynom
    that projects incident angle`theta` to radius
    `(x, y, z)` is the view vector with angle`theta` to `(0, 0, 1)`
    radius is the length of `(x, y)` vector
    '''
    r = torch.linalg.norm(image_points - principal_point, dim=-1)
    z = polyval(r, poly)
    theta = torch.arctan(r / z)
    from numpy.polynomial import Polynomial

    # leave out absolute term
    degree = torch.arange(len(poly) - 1) + 1
    poly_theta = Polynomial.fit(theta.cpu().flatten().numpy(), r.cpu(
    ).flatten().numpy(), deg=degree.tolist(), domain=(-1, 1)).coef
    return torch.from_numpy(poly_theta).to(poly)


def poly_theta_to_r(poly: Tensor, view_points: Tensor) -> Tensor:
    '''
    Convert polynom that projects incident angle`theta` to radius to polynom
    that projects radius to `z`
    `(x, y, z)` is the view vector with angle`theta` to `(0, 0, 1)`
    radius is the length of `(x, y)` vector
    '''
    theta = get_theta(view_points)
    r = polyval(theta, poly)
    z = r / torch.tan(theta)
    from numpy.polynomial import Polynomial

    # leave out linear term
    degree = torch.arange(len(poly) - 1)
    degree[1:] += 1
    poly_r = Polynomial.fit(r.cpu().flatten().numpy(), z.cpu(
    ).flatten().numpy(), deg=degree.tolist(), domain=(-1, 1)).coef
    return torch.from_numpy(poly_r).to(poly)
