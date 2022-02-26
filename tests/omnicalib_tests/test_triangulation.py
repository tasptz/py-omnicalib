import torch
from autograd import numpy as np
from omnicalib.stereo import get_midpoint, project_points, optimize_midpoint
from omnicalib.polyfit import poly_theta_to_r, polyval
from omnicalib.geometry import unit


def test_midpoint():
    targets = torch.tensor((0.5, 0, 1.0))[None]
    targets = targets.repeat(10, 1)
    targets[:, 1] = torch.rand(10) - 0.5

    view_origin = targets.new_tensor((1, 0, 0))
    view_z = unit(targets.mean(dim=0) - view_origin)
    view_y = unit(targets.new_tensor((0, 1, 0)) + targets.new_empty(3).uniform_(-0.1, 0.1))
    view_x = unit(torch.cross(view_y, view_z))
    R = torch.stack((view_x, view_y, view_z))

    v0 = targets
    v1 = (targets - view_origin) @ R.t()

    transform = torch.cat((R, -R @ view_origin[:, None]), dim=1)
    assert torch.allclose(targets, get_midpoint(v0, v1, transform), atol=0.1, rtol=0.)


def test_optimize_midpoint():
    def radius(focal_length, theta):
        '''equisolid'''
        return 2 * focal_length * torch.sin(theta / 2)

    focal_length = 500.
    theta = torch.linspace(1e-6, torch.pi / 4, 100, dtype=torch.float64)
    r = radius(focal_length, theta)
    M = torch.stack((theta, torch.pow(theta, 2), torch.pow(theta, 3)), dim=1)
    poly_theta = torch.linalg.lstsq(M, r)[0]
    poly_theta = torch.cat((poly_theta.new_zeros(1), poly_theta))

    principal_point = theta.new_tensor((1024, 1024)) / 2 + 0.5

    target = theta.new_tensor((0.5, 0, 1))

    view_origin = target.new_tensor((1, 0, 0))
    view_z = unit(target - view_origin)
    view_y = unit(target.new_tensor((0.5, 0.5, 0)))
    view_x = unit(torch.cross(view_y, view_z))
    R = torch.stack((view_x, view_y, view_z))
    transform = torch.cat((
        R,
        R @ -view_origin.view(3, 1)
    ), dim=1)

    projected_point0 = torch.from_numpy(project_points(
        poly_theta.numpy(),
        principal_point.numpy(),
        np.eye(4, dtype=np.float64),
        target.numpy())).squeeze()
    projected_point1 = torch.from_numpy(project_points(
        poly_theta.numpy(),
        principal_point.numpy(),
        transform.numpy(),
        target.numpy())).squeeze()

    # from matplotlib import pyplot as plt
    # fig, ax = plt.subplots(2, subplot_kw={'box_aspect': 1})
    # for i, pp in enumerate([projected_point0, projected_point1]):
    #     ax[i].scatter(pp[0], pp[1], marker='+', color='g')

    N = 100
    projected_point0 = projected_point0[None] + \
        projected_point0.new_empty(N, 2).normal_(std=1)
    projected_point1 = projected_point1[None] + \
        projected_point1.new_empty(N, 2).normal_(std=1)

    # for i, pp in enumerate([projected_point0, projected_point1]):
    #     ax[i].scatter(pp.t()[0], pp.t()[1], marker='x', color='r')
    #     ax[i].set_xlim(0, 1024)
    #     ax[i].set_ylim(0, 1024)
    # plt.show()

    poly_r = poly_theta_to_r(
        poly_theta.squeeze(0),
        torch.stack((
            torch.sin(theta),
            torch.zeros_like(theta),
            torch.cos(theta)
        ), dim=1)
    )

    p0 = projected_point0 - principal_point
    z0 = polyval(torch.sqrt(torch.pow(p0, 2).sum(dim=1)), poly_r)
    p1 = projected_point1 - principal_point
    z1 = polyval(torch.sqrt(torch.pow(p1, 2).sum(dim=1)), poly_r)

    vectors0 = torch.cat((p0, z0[:, None]), dim=-1)
    vectors1 = torch.cat((p1, z1[:, None]), dim=-1)

    midpoints = optimize_midpoint(
        poly_theta,
        principal_point,
        vectors0,
        projected_point0,
        poly_theta,
        principal_point,
        vectors1,
        projected_point1,
        transform
    )
    midpoint = midpoints.mean(dim=0)
    assert torch.allclose(target, midpoint, atol=1e-2)

if __name__ == '__main__':
    test_optimize_midpoint()
