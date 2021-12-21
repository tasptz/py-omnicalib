from typing import List, Tuple

import torch
from torch import Tensor

from omnicalib.geometry import gram_schmidt, unit


def solve_scale(b3: Tensor, R: Tensor) -> Tuple[Tensor, Tensor]:
    '''
    From a scaled `3 x 2` submatrix of a `3 x 3` rotation matrix
    solve for the unscaled `3 x 2` submatrix
    '''
    R = torch.cat((R, R.new_zeros((1, 2))))
    R[2, 1] = b3
    scale = 1. / torch.linalg.norm(R[:, 1])
    R = R * scale
    v = torch.pow(R[:2, 0], 2).sum()
    if v > 1.:
        raise ValueError()
    R[2, 0] = torch.sqrt(1. - v)
    return R, scale


def solve_full(R: Tensor) -> Tensor:
    '''
    Construct the 3rd column of `3 x 2` sub matrix
    of a `3 x 3` rotation matrix
    '''
    R = gram_schmidt(R[None])[0]
    r3 = unit(torch.cross(R[:, 0], R[:, 1], dim=-1)).view(3, 1)
    R = torch.cat((R, r3), dim=1)
    assert torch.allclose(torch.linalg.det(R), R.new_ones(1))
    return R


def solve_r21(R: Tensor) -> Tensor:
    '''
    Given a scaled `2 x 2` submatrix of a `3 x 3` rotation matrix
    solve for the scaled third element of the second column: `r21`
    '''
    # a = r1
    a1 = R[0, 0]
    a2 = R[1, 0]
    # b = r2
    b1 = R[0, 1]
    b2 = R[1, 1]

    p = torch.pow(b1, 2) + torch.pow(b2, 2) - \
        torch.pow(a1, 2) + torch.pow(a2, 2)
    q = -torch.pow(a1 * b1 + a2 * b2, 2)

    u = -p / 2
    v = torch.sqrt(torch.pow(p / 2, 2) - q)
    b3_0 = torch.sqrt(u + v)
    # b3_1 = torch.sqrt(u - v)

    b3_solutions = b3_0.view(1)
    # b3_solutions = torch.stack((b3_0, -b3_0, b3_1, -b3_1))
    return b3_solutions[~torch.isnan(b3_solutions)]


def orthonorm(rot: Tensor, trans: Tensor) -> List[Tuple[Tensor, Tensor]]:
    '''
    Given the scaled `2 x 2` submatrix of a `3 x 3` rotation matrix
    and the scaled `(x, y)` components of a 3d translation vector
    solve for the full `3 x 3` rotation matrix `R` and 3d translation
    vector `t`

    Note: the `2 x 2` submatrix can be permuted before solving,
    which results in different solutions
    '''
    solution = []
    for r, t in ((rot, trans), (-rot, -trans)):
        for b3 in solve_r21(r):
            try:
                rs, scale = solve_scale(b3, r)
                solution.append((
                    solve_full(rs),
                    t * scale
                ))
            except ValueError:
                pass
        for b3 in solve_r21(r.t()):
            try:
                rs_t, scale = solve_scale(b3, r)
                solution.append((
                    solve_full(rs_t).t(),
                    t * scale
                ))
            except ValueError:
                pass
        r_flip = torch.flip(r, dims=(1,))
        for b3 in solve_r21(r_flip):
            try:
                rs_flipped, scale = solve_scale(b3, r_flip)
                solution.append((
                    solve_full(torch.flip(rs_flipped, dims=(1,))),
                    t * scale
                ))
            except ValueError:
                pass
        r_t_flip = torch.flip(r.t(), dims=(1,))
        for b3 in solve_r21(r_t_flip):
            try:
                rs_t_flipped, scale = solve_scale(b3, r_t_flip)
                solution.append((
                    solve_full(torch.flip(rs_t_flipped, dims=(1,))).t(),
                    t * scale
                ))
            except ValueError:
                pass
    return solution
