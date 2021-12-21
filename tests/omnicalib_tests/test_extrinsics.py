import torch

from omnicalib.extrinsics import partial_extrinsics
from . import get_data


def test_partial_extrinsics():
    R, T, world_points, _view_points, image_points = get_data(8, 'equidistant')

    R_test, T_test = partial_extrinsics(image_points, world_points)
    # unknown scale factor in solution!
    s = R[:, 0, 0] / R_test[:, 0, 0]
    R_test *= s[:, None, None]
    T_test *= s[:, None]
    assert torch.allclose(R[:, :2, :2], R_test, atol=1e-2)
    # tolerance 1 mm
    assert torch.allclose(T[:, :2], T_test, atol=1)
