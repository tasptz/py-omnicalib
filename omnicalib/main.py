import logging
from typing import Dict

import cv2 as cv
import torch
from matplotlib import pyplot as plt
from torch import Tensor

from .calibrate import calibrate
from .geometry import get_theta, transform
from .plot import plot_rz_curve, plot_thetar_curve

logging.basicConfig(level=logging.INFO)


def main(detections: Dict, degree: int, threshold: float, count: int,
         principal_point: Tensor = None,
         spiral_step: int = 10, spiral_end: int = 100):
    '''
    Full calibration with plots
    '''
    H, W = cv.imread(str(next(iter(detections.keys()))),
                     cv.IMREAD_GRAYSCALE).shape
    scale_image = 512 / max(W, H)
    image_paths = sorted(detections.keys())
    image_points = torch.stack(
        [detections[k]['image_points'] for k in image_paths]).to(torch.float64)
    world_points = torch.stack(
        [detections[k]['object_points'] for k in image_paths]).to(torch.float64)
    images = torch.stack([
        torch.from_numpy(cv.cvtColor(
            cv.resize(cv.imread(str(k)), None, fx=scale_image, fy=scale_image),
            cv.COLOR_BGR2RGB
        ))
        for k in image_paths
    ])

    search_principal_point = principal_point is None
    image_center = image_points.new_tensor((W, H)) * 0.5 - 0.5
    if principal_point is not None:
        principal_point = image_points.new_tensor(
            principal_point) + image_center
    else:
        principal_point = image_center

    try:
        R, T, poly_theta, poly, principal_point = calibrate(
            degree,
            threshold,
            count,
            image_points,
            world_points,
            principal_point,
            images,
            (H, W),
            spiral_step,
            spiral_end
        )
        with open('calibration.yml', 'w') as f:
            data = {
                'extrinsics': torch.cat((R, T[..., None]), dim=2).tolist(),
                'poly_incident_angle_to_radius': poly_theta.tolist(),
                'poly_radius_to_z': poly.tolist(),
                'principal_point': principal_point.tolist()
            }
            import yaml
            yaml.dump(data, f, Dumper=yaml.SafeDumper)
        norm_radius = min(H, W) * 0.5 - 0.5
        min_dim_str = 'width' if W <= H else 'height'
        fig = plot_rz_curve(
            torch.linalg.norm(image_points - principal_point, dim=-1).max(),
            poly,
            norm_radius,
            xlabel=f'Radius / Image {min_dim_str}',
            ylabel=f'Z / Image {min_dim_str}'
        )
        fig.savefig('rz_curve.pdf', dpi=300)
        vp = transform(world_points, R, T)
        fig = plot_thetar_curve(
            get_theta(vp).max(),
            poly_theta,
            norm_radius,
            ylabel=f'Radius / Image {min_dim_str}'
        )
        fig.savefig('thetar_curve.pdf', dpi=300)
        plt.show()
    except Exception as e:
        import sys
        sys.stderr.write(str(e))
        raise e


if __name__ == '__main__':
    main()
