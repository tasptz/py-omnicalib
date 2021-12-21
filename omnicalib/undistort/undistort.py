from typing import Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from ..geometry import unit
from ..polyfit import polyval
from ..projection import project_poly_thetar


def image_to_view(poly_rz, principal_point, p_image: Tensor,
                  norm: bool = False) -> Tensor:
    xy = p_image[..., :2] - principal_point
    radius = torch.linalg.norm(
        xy,
        dim=-1,
        keepdim=True
    )
    v_view = torch.cat((
        xy,
        polyval(radius, poly_rz)
    ), dim=-1)
    return unit(v_view) if norm else v_view


class Projection:
    def __init__(self, poly_thetar: Tensor, poly_rz: Tensor,
                 principal_point: Tensor):
        self.poly_thetar = poly_thetar
        self.poly_rz = poly_rz
        self.principal_point = principal_point

    def image_to_view(self, p_image: Tensor, norm: bool = False) -> Tensor:
        return image_to_view(
            self.poly_rz,
            self.principal_point,
            p_image,
            norm
        )

    def view_to_image(self, v_view: Tensor, normed: bool = False) -> Tensor:
        return project_poly_thetar(
            v_view,
            self.poly_thetar,
            self.principal_point,
            normed
        )


def get_view_vectors(image_shape: Tuple[int, int], fovx: Tensor) -> Tensor:
    H, W = image_shape
    focal_length = (W / 2) / torch.tan(fovx)
    gx, gy = torch.meshgrid(torch.arange(
        0, W), torch.arange(0, H), indexing='xy')
    gx = gx.to(fovx)
    gy = gy.to(fovx)
    g = torch.stack((gx + 0.5 - W * 0.5, gy + 0.5 - H * 0.5,
                    torch.empty_like(gy).fill_(focal_length)), dim=-1)
    return unit(g).view(-1, 3)


def undistort(poly_thetar: Tensor, poly_rz: Tensor,
              principal_point: Tensor, view_image: Tensor,
              view_shape: Tuple[int, int],
              image: Tensor, p_image: Tensor,
              down: Tensor = None,
              A_view: Tensor = None,
              mode: str = 'bilinear'):
    view = image_to_view(poly_rz, principal_point, p_image, True)
    if down is None:
        down = unit(p_image - principal_point)
    down = down.expand_as(view)
    right = unit(torch.cross(down, view, dim=-1))
    down = unit(torch.cross(view, right))
    M = torch.stack((right, down, view), dim=-1)
    if A_view is not None:
        M = M @ A_view
        normalize = torch.allclose(torch.linalg.det(
            A_view).abs(), A_view.new_ones(1), atol=1e-3)
    else:
        normalize = False
    view_transformed = (view_image[None, :, None, :]
                        @ M[:, None].swapaxes(-2, -1)).squeeze(-2)
    if normalize:
        view_transformed = unit(view_transformed)
    p_img = project_poly_thetar(
        view_transformed,
        poly_thetar,
        principal_point,
        True
    )

    H, W = image.shape[-2:]
    p_rel = (p_img + 0.5) / p_img.new_tensor((W, H)) * 2 - 1

    Hv, Wv = view_shape
    return M, F.grid_sample(
        image.expand(p_rel.shape[0], -1, -1, -1),
        p_rel.view(-1, Hv, Wv, 2),
        mode=mode,
        align_corners=False
    )


class Undistort:
    def __init__(self, projection: Projection, view_shape: Tuple[int, int],
                 fovx: Tensor, down: Tensor = None, mode: str = 'bilinear'):
        self.projection = projection
        self.view_shape = view_shape
        self.view_vectors = get_view_vectors(view_shape, fovx)
        self.down = down
        self.mode = mode

    def __call__(self, image: Tensor, p_image: Tensor, down: Tensor = None,
                 A_view: Tensor = None) -> Tuple[Tensor, Tensor]:
        return undistort(
            self.projection.poly_thetar,
            self.projection.poly_rz,
            self.projection.principal_point,
            self.view_vectors,
            self.view_shape,
            image,
            p_image,
            self.down if down is None else down,
            A_view,
            self.mode
        )
