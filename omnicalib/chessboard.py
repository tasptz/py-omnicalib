import torch
from torch import Tensor


def get_points(rows: int, cols: int, size: float) -> Tensor:
    '''
    Generate the world points of a chessboard marker
    matching OpenCV conventions
    '''
    x, y = torch.meshgrid(torch.arange(
        cols), torch.arange(rows), indexing='xy')
    return torch.stack((x, y, torch.zeros_like(y)), dim=2).to(torch.float64) \
        * size
