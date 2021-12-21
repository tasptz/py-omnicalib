import torch
from omnicalib.geometry import unit
from omnicalib.polyfit import polyfit, polynom

from . import get_data


def test_polyfit():
    R, T, world_points, view_points, image_points = get_data(8, 'equidistant')

    degree = 4
    poly, T_test = polyfit(degree, image_points, world_points, R, T[:, :2])

    # tolerance 5 mm
    assert torch.allclose(T, T_test, atol=5)

    rho = torch.linalg.norm(image_points[..., :2], dim=-1)

    z_test = polynom(rho, degree) @ poly[None, :, None]
    view_vec_test = unit(torch.cat((image_points, z_test), dim=-1))
    view_vec = unit(view_points)

    dot = (view_vec_test * view_vec).sum(dim=-1)
    assert torch.allclose(dot, torch.ones_like(dot), atol=1e-2)
