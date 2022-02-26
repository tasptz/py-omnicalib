'''
Detect chessboard marker in images and save corner points in file
'''
import pickle
from itertools import combinations
from pathlib import Path
from typing import List

import torch
import yaml
from torch import Tensor

from ..extrinsics import full_extrinsics
from .reprojection import project_points, reprojection


def get_extrinsics(calibrations: dict, image_points: Tensor,
                   object_points: Tensor) -> Tensor:
    matrices = []
    for c, i, o in zip(calibrations, image_points, object_points):
        poly = torch.tensor(c['poly_radius_to_z'], dtype=torch.float64)
        pp = torch.tensor(c['principal_point'], dtype=torch.float64)
        R, T = full_extrinsics(
            poly,
            i - pp,
            o
        )
        B = len(R)
        M = torch.eye(4, dtype=torch.float64)[None].tile(B, 1, 1)
        M[:, :3, :3] = R
        M[:, :3, 3] = T
        matrices.append(M)
    return torch.stack(matrices)


def plot_all(poly_thetas: Tensor, principal_points: Tensor,
             image_points: Tensor, object_points: Tensor,
             extrinsics: Tensor, image_paths: List[List[Path]]) -> None:
    projected_points = project_points(
        poly_thetas.numpy()[:, None, None],
        principal_points.numpy()[:, None, None],
        extrinsics.numpy(),
        object_points.numpy()
    )

    num_cam, num_img = extrinsics.shape[:2]

    from matplotlib import pyplot as plt
    from matplotlib.image import imread
    fig, ax = plt.subplots(num_cam, num_img, squeeze=False)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.01)
    for c in range(num_cam):
        for i in range(num_img):
            pp = projected_points[c, i].T
            ip = image_points[c, i].t()
            a = ax[c, i]
            a.imshow(imread(image_paths[c][i]))
            a.scatter(pp[0], pp[1], color='r', marker='+')
            a.scatter(ip[0], ip[1], color='g', marker='x')
            a.set_title(image_paths[c][i].stem)
            a.axis('off')
    plt.show()


def calibrate_stereo(calibrations: dict, image_points: Tensor,
                     object_points: Tensor,
                     image_paths: List[List[Path]]) -> None:
    '''
    calibrations
    image_points (num_cameras, num_images, num_points, 2)
    object_points (..., 3)
    '''
    extrinsics = get_extrinsics(calibrations, image_points, object_points)
    poly_thetas = torch.stack(
        [torch.tensor(c['poly_incident_angle_to_radius'], dtype=torch.float64)
         for c in calibrations])
    principal_points = torch.stack(
        [torch.tensor(c['principal_point'], dtype=torch.float64)
         for c in calibrations])

    plot_all(poly_thetas, principal_points, image_points,
             object_points, extrinsics, image_paths)

    for c0, c1 in combinations(range(len(extrinsics)), 2):
        for i in range(image_points.shape[1]):
            # optimize 2 cameras on one image pair
            e, o = reprojection(
                extrinsics[[c0, c1], i: i + 1],
                poly_thetas[:2, None, None],
                principal_points[:2, None, None],
                image_points[[c0, c1], i: i + 1],
                object_points[[c0, c1], i: i + 1]
            )
            extrinsics[c0, i] = e.squeeze()
            extrinsics[c1, i] = o.squeeze() @ e.squeeze()

    e, o = reprojection(
        extrinsics,
        poly_thetas[:, None, None],
        principal_points[:, None, None],
        image_points,
        object_points
    )
    extrinsics[0, :] = e
    extrinsics[1:] = o[:, None] @ e[None]

    plot_all(poly_thetas, principal_points, image_points,
             object_points, extrinsics, image_paths)
    with open('stereo.yml', 'w') as f:
        yaml.dump({
            'calibrations': calibrations,
            'relative_transforms': {
                f'0 to {i + 1}': x.tolist() for i, x in enumerate(o)
            }
        }, f, Dumper=yaml.SafeDumper)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--calibrations', type=str,
                        nargs='+', required=True)
    parser.add_argument('-d', '--detections', type=str,
                        nargs='+', required=True)
    args = parser.parse_args()

    calibrations = []
    for fn in args.calibrations:
        with open(fn, 'r') as f:
            calibrations.append(yaml.load(f, Loader=yaml.SafeLoader))

    image_paths = []
    image_points = []
    object_points = []
    for fn in args.detections:
        with open(fn, 'rb') as f:
            data = pickle.load(f)
        detections = [
            (k,) + tuple(v[k_data]
                         for k_data in ('image_points', 'object_points'))
            for k, v in data['detections'].items()
        ]
        detections = sorted(detections, key=lambda x: x[0])
        ipa, ipo, opo = zip(*detections)
        image_paths.append(ipa)
        image_points.append(torch.stack(ipo))
        object_points.append(torch.stack(opo))

    image_points = torch.stack(image_points)
    object_points = torch.stack(object_points)
    calibrate_stereo(
        calibrations,
        image_points,
        object_points,
        image_paths
    )


if __name__ == '__main__':
    main()
