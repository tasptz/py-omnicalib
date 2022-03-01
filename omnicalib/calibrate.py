import logging
from typing import Tuple

import torch
from torch import Tensor
from tqdm import tqdm

from . import optim
from .extrinsics import full_extrinsics, partial_extrinsics
from .geometry import check_origin, spiral, transform
from .orthonorm import orthonorm
from .plot import Scatter
from .polyfit import poly_theta_to_r, polyfit
from .projection import project_poly_rz, project_poly_thetar


def get_reprojection_error(poly: Tensor, r: Tensor, t: Tensor, ip: Tensor,
                           wp: Tensor) -> Tensor:
    '''
    Reprojection error per image point
    '''
    view_points = transform(wp, r, t)
    ip_c = project_poly_rz(view_points, poly)
    reprojection_error = torch.linalg.norm(ip - ip_c, dim=-1)
    return reprojection_error


def fit_reprojection_error(degree: int, r: Tensor, t_par: Tensor, ip: Tensor,
                           wp: Tensor) -> float:
    '''
    Fit polynomial and calculate mean reprojection error
    '''
    poly, t = polyfit(degree, ip, wp, r, t_par)
    # reject solution
    if poly[0] < 0 or not check_origin(r, t).squeeze().item():
        return float('inf')
    return get_reprojection_error(poly, r, t, ip, wp).squeeze()


def show_points(title: str, figure_path: str, image_points: Tensor,
                projected_points: Tensor,
                images=None, image_shape=None) -> None:
    '''
    Scatter plot of image points and reprojected points
    '''
    scatter = Scatter(title, len(image_points))
    if images is not None:
        scatter.imshow(images, image_shape)
    scatter(image_points, color='g', marker='o')
    scatter(projected_points, color='r', marker='x')
    scatter.save(f'{figure_path}.pdf', dpi=300)
    scatter.show()


def _latex_float(v):
    '''
    Format number in latex math syntax
    '''
    s = f'{v:.1e}'
    if s.endswith('e+00'):
        return s[:-4]
    elif s == 'nan':
        return '\\mathrm{nan}'
    else:
        base, exponent = s.split('e')
        base, exponent = float(base), int(exponent)
        return f'{base}\\mathrm{{e}}{{{exponent:+d}}}'


def get_error_str(x):
    '''
    Format error to nice string
    '''
    x = x.flatten()
    return (
        f'reprojection error $\\mu={_latex_float(x.mean())},'
        f' \\sigma={_latex_float(x.std() if len(x) > 1 else 0.)}$'
    )


def calibrate(degree: int, reprojection_error_threshold: float,
              reprojection_count: int, image_points: Tensor,
              world_points: Tensor, principal_point_initial: Tensor,
              images: Tensor = None,
              image_shape: Tuple[int, int] = None,
              spiral_step: int = 10, spiral_end: int = 100) \
        -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    '''
    Full calibration algorithm
    '''
    logger = logging.getLogger('calibrate')

    if spiral_step and spiral_end:
        logger.info(
            'Brute-force search for principal point'
            f' ({reprojection_count} images'
            ' with mean reprojection error <='
            f' {reprojection_error_threshold:.1f})'
        )
        progress = tqdm(list(spiral(spiral_step, end=spiral_end)))
    else:
        progress = tqdm([principal_point_initial])

    for x_spiral, y_spiral in progress:
        principal_point = principal_point_initial + \
            image_points.new_tensor((x_spiral, y_spiral))
        image_points_spiral = image_points - principal_point

        R_par, T_par = partial_extrinsics(image_points_spiral, world_points)

        valid = []
        results = []
        for idx, (r_par, t_par) in enumerate(zip(R_par, T_par)):
            ip = image_points_spiral[idx]
            wp = world_points[idx]
            min_error = float('inf')
            best = None
            for r_ort, t_par_ort in orthonorm(r_par, t_par):
                e = fit_reprojection_error(
                    degree, r_ort[None], t_par_ort[None], ip[None], wp[None])
                e_mean = e.mean().item() if isinstance(e, Tensor) else e
                if e_mean < min_error:
                    min_error = e_mean
                    if e_mean < reprojection_error_threshold:
                        best = e, r_ort, t_par_ort
            if best is None:
                valid.append(False)
            else:
                valid.append(True)
                results.append(best)
        progress.set_description(
            f'{min_error:.3f} {int(x_spiral):+4d} {int(y_spiral):+4d}')
        if len(results) >= reprojection_count:
            break

    valid = torch.tensor(valid)
    logger.info(f'Valid solution for {valid.sum()}/{len(valid)} images')
    if not torch.any(valid):
        raise Exception('No initial solution found')

    reprojection_errors, R, T_par_ort = [torch.stack(x) for x in zip(*results)]
    assert torch.allclose(torch.linalg.det(R), R.new_ones(len(R)))
    mean_reprojection_errors = reprojection_errors.mean(dim=1)
    reprojection_error_threshold = \
        torch.sort(mean_reprojection_errors)[0][reprojection_count - 1]
    fit_mask = mean_reprojection_errors <= reprojection_error_threshold

    error_str = get_error_str(reprojection_errors[fit_mask])
    logger.info(f'Initial {error_str}')
    logger.info(
        f'Initial principal point ({principal_point[0]:.1f},'
        f' {principal_point[1]:.1f})'
    )

    logger.info(
        f'Initial solution for {fit_mask.sum()}/{len(fit_mask)} selected'
    )
    valid[torch.where(valid)[0][~fit_mask]] = False
    R = R[fit_mask]
    assert torch.allclose(torch.linalg.det(R), R.new_ones(len(R)))
    poly, T = polyfit(degree, image_points[valid] - principal_point,
                      world_points[valid], R, T_par_ort[fit_mask]
                      )
    show_points(
        f'Initial Solution for Subset ({error_str})',
        'initial_solution',
        image_points[valid],
        project_poly_rz(
            transform(world_points[valid], R, T), poly, principal_point),
        images[valid] if images is not None else None,
        image_shape
    )
    assert torch.allclose(torch.linalg.det(R), R.new_ones(len(R)))
    success, reprojection_errors, res = optim.reprojection(
        R, T, poly, principal_point, image_points[valid],
        world_points[valid]
    )
    if not success:
        raise Exception('Optmization failed')

    error_str = get_error_str(reprojection_errors)
    logger.info(f'Optimized {error_str}')
    R, T, poly_theta, principal_point = res
    assert torch.allclose(torch.linalg.det(R), R.new_ones(len(R)))
    logger.info(
        f'Optimized principal point ({principal_point[0]:.1f},'
        f' {principal_point[1]:.1f})'
    )
    show_points(
        f'Optimized Solution for Subset ({error_str})',
        'optimized_solution',
        image_points[valid],
        project_poly_thetar(
            transform(world_points[valid], R, T), poly_theta, principal_point),
        images[valid] if images is not None else None,
        image_shape
    )

    poly = poly_theta_to_r(poly_theta, transform(world_points[valid], R, T))

    R, T = full_extrinsics(poly, image_points - principal_point, world_points)
    assert torch.allclose(torch.linalg.det(R), R.new_ones(len(R)), atol=1e-2)

    success, reprojection_errors, res = optim.reprojection(
        R, T, poly, principal_point, image_points, world_points)
    if not success:
        raise Exception('Optmization failed')
    error_str = get_error_str(reprojection_errors)
    logger.info(f'Final {error_str}')
    R, T, poly_theta, principal_point = res
    assert torch.allclose(torch.linalg.det(R), R.new_ones(len(R)), atol=1e-2)
    assert torch.all(check_origin(R, T))

    logger.info(
        f'Final principal point ({principal_point[0]:.1f},'
        f' {principal_point[1]:.1f})'
    )
    show_points(
        f'Final Solution ({error_str})',
        'final_solution',
        image_points,
        project_poly_thetar(transform(world_points, R, T),
                            poly_theta, principal_point),
        images,
        image_shape
    )
    poly = poly_theta_to_r(poly_theta, transform(world_points, R, T))
    return R, T, poly_theta, poly, principal_point
