from pathlib import Path
from typing import Callable, Tuple, Union

import torch
from matplotlib import pyplot as plt
from torch import Tensor

from .polyfit import polyval


class Scatter:
    '''
    Helper class for scatter plots of image points and reprojected points
    '''

    def __init__(self, title: str, num_plots: int, dim: int = None):
        import math
        cols = int(math.ceil(math.sqrt(num_plots)))
        rows = int(math.ceil(num_plots / cols))
        self.fig, ax = plt.subplots(
            rows, cols, squeeze=False, figsize=(12, 12))
        plt.suptitle(title)
        ax = ax.flatten()
        for i in range(num_plots, rows * cols):
            ax[i].axis('off')
        self.ax = ax[:num_plots]
        for a in self.ax:
            a.axis('equal')
            a.grid(True)
        if dim is not None:
            from matplotlib.patches import Rectangle
            for a in self.ax:
                a.add_patch(Rectangle((-dim, -dim), 2 *
                            dim, 2 * dim, fill=False))

    def imshow(self, images: Tensor, image_shape: Tuple[int]) -> None:
        '''
        Show images with arbitrary resolution matched to `image_shape`
        '''
        self.fig.subplots_adjust(0.01, 0.01, 0.99, 0.90, 0.0, 0.0)
        H, W = image_shape
        for a, i in zip(self.ax, images):
            a.imshow(i, extent=(-0.5, W - 0.5, H - 0.5, -0.5))
            a.axis('off')

    def __call__(self, points: Tensor, **kwargs):
        '''
        Scatter plot given points
        '''
        for a, p in zip(self.ax, points):
            a.scatter(p[:, 0], p[:, 1], **kwargs)

    def show(self) -> None:
        '''
        Show the plot
        '''
        plt.show()

    def save(self, filepath: Union[str, Path], *args, **kwargs):
        '''
        Save the figure to `filepath`
        '''
        self.fig.savefig(filepath, *args, **kwargs)

    def clear(self) -> None:
        '''
        Clear all axes
        '''
        for a in self.ax:
            a.clear()

    def close(self) -> None:
        '''
        Close the figure
        '''
        plt.close(self.fig)


def plot_rz_curve(r_max: float, poly: Tensor, norm: float,
                  num_points: int = 100,
                  xlabel: str = None, ylabel: str = None,
                  f_ideal: Callable[[Tensor], Tensor] = None) -> plt.Figure:
    '''
    Plots the radius (image points) over `z` coordinate
    Radius is the length of the `(x, y)` vector
    `(x, y, z)` forms the view vector into the view coordinate system
    '''
    x = torch.linspace(0, r_max, num_points).to(poly)
    y = polyval(x, poly)
    fig = plt.figure()
    plt.plot(x / norm, y / norm, color='r', label='calibrated')
    if f_ideal is not None:
        plt.plot(x / norm, f_ideal(x) / norm, color='g', label='ideal')
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel(xlabel or 'Radius in relative coordinates')
    plt.ylabel(ylabel or 'Z in relative coordinates')
    plt.legend()
    return fig


def plot_thetar_curve(theta_max: float, poly: Tensor, norm: float,
                      num_points: int = 100,
                      xlabel: str = None, ylabel: str = None,
                      f_ideal: Callable[[Tensor], Tensor] = None) \
        -> plt.Figure:
    '''
    Plots the incident angle `theta` over the radius
    Given a view vector `(x, y, z)`, `theta` is the angle between
    this vector and `(0, 0, 1)`
    Radius is the length of the `(x, y)` vector
    '''
    x = torch.linspace(0, theta_max, num_points).to(poly)
    y = polyval(x, poly) / norm
    fig = plt.figure()
    plt.plot(x, y, color='r', label='calibrated')
    if f_ideal is not None:
        plt.plot(x, f_ideal(x) / norm, color='g', label='ideal')
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel(xlabel or 'Incident angle in radian')
    plt.ylabel(ylabel or 'Radius in relative coordinates')
    plt.legend()
    return fig
