import torch
from omnicalib.orthonorm import orthonorm

from . import get_data, get_generator


def test_orthonorm():
    R, T, _world_points, _view_points, image_points = get_data(
        8, 'equidistant')

    g = get_generator()

    scale = torch.ones(1).uniform_(1e-6, 2, generator=g).item()

    def test_dot(r, r_test):
        p = r * r_test
        dot_col = p.sum(dim=0).abs().min()
        dot_row = p.sum(dim=1).abs().min()
        return 1 - dot_col < 1e-2 and 1 - dot_row < 1e-2

    for r, t in zip(R, T):
        R_test, T_test = [torch.stack(x) for x in zip(
            *orthonorm(r[:2, :2] * scale, t[:2] * scale))]
        det = torch.linalg.det(R_test)
        assert torch.allclose(det, torch.ones_like(det))
        for r_test, t_test in zip(R_test, T_test):
            # tolerance 1 mm
            if torch.allclose(t[:2], t_test, atol=1) and test_dot(r, r_test):
                return
    raise Exception()
