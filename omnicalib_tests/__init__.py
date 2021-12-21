from typing import Tuple

import torch
from omnicalib.chessboard import get_points
from omnicalib.geometry import transform
from omnicalib.simulation import get_view_matrix, project_ideal
from torch import Tensor

SEED = 2147483643


def get_generator(seed: int = None) -> torch.Generator:
    if seed is None:
        seed = SEED
    g = torch.Generator()
    g.manual_seed(SEED)
    return g


def get_data(focal_length: float, model: str, n: int = 30,
             noise: float = 0.05) \
        -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    g = get_generator()
    origin = torch.empty(
        (n, 3), dtype=torch.float64).uniform_(-0.05, 0.05, generator=g)
    origin[:, 2] -= 0.3
    look_at = origin.new_empty((n, 3)).uniform_(-0.05, 0.05, generator=g)
    down = origin.new_empty((n, 3)).uniform_(-1, 1, generator=g)
    down[:, 2] *= 0.05

    R, T = get_view_matrix(origin * 1e3, look_at * 1e3, down * 1e3)

    world_points = get_points(6, 8, 100.).view(-1, 3)
    world_points -= world_points.mean(dim=0, keepdim=True)
    world_points = world_points.view(1, -1, 3).expand(len(R), -1, -1)

    view_points = transform(world_points, R, T)

    image_points = project_ideal(focal_length, view_points, model)

    # noise
    image_points = image_points + \
        torch.empty_like(image_points).uniform_(-noise, noise, generator=g)

    return R, T, world_points, view_points, image_points
