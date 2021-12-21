'''
Detect chessboard marker in images and save corner points in file
'''
import math
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import cv2 as cv
import torch
from autograd import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from tqdm import tqdm

from ..chessboard import get_points


def scale_corners(corners: np.ndarray, shape_in: Tuple[int],
                  shape_out: Tuple[int]) -> np.ndarray:
    '''
    Scale corners to different resolution
    '''
    sin, sout = [
        np.array(x[::-1], dtype=corners.dtype)
        for x in (shape_in, shape_out)
    ]
    return (corners + 0.5) / sin * sout - 0.5


def get_corners(img_path: Path, chessboard_shape: Tuple[int], max_dim: int,
                **_kwargs) -> Tuple[Path, np.ndarray]:
    '''
    Detect corners for image at `img_path`
    '''
    try:
        img = img_corners = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
        H, W = img.shape
        if max_dim and max_dim < (m := max(H, W)):
            def f(D):
                return int(round(D * max_dim / m))
            img_corners = cv.resize(img, (f(W), f(H)))
        retval, corners = cv.findChessboardCorners(
            img_corners,
            chessboard_shape[::-1],
            cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
        )
        if not retval:
            retval, corners = cv.findChessboardCornersSB(
                img_corners,
                chessboard_shape[::-1],
                cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_EXHAUSTIVE
            )
        if retval:
            if img.shape != img_corners.shape:
                corners = scale_corners(corners, img_corners.shape, img.shape)
            win_size = (5, 5)
            zero_zone = (-1, -1)
            criteria = (
                cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT,
                40,
                0.001
            )
            corners = cv.cornerSubPix(img, corners, win_size, zero_zone,
                                      criteria).squeeze(1)
        else:
            corners = None
        return img_path.resolve(), corners
    except Exception as e:
        import sys
        sys.stderr.write(str(e))


def detect(image_path: Path, threads: int, **kwargs):
    '''
    Detect corners in all images at `image_path`
    '''
    results = {}
    with ThreadPoolExecutor(threads) as executor:
        futures = [
            executor.submit(get_corners, p, **kwargs)
            for p in sorted(image_path.iterdir())
        ]
        failed = 0
        progress = tqdm(as_completed(futures), total=len(futures))
        for x in progress:
            p, corners = x.result()
            results[p] = corners
            if corners is None:
                failed += 1
            progress.set_description(f'failed={failed / len(results):.0%}')
    with open('detections.pickle', 'wb') as f:
        chessboard = get_points(
            *kwargs['chessboard_shape'],
            kwargs['chessboard_size']
        ).view(-1, 3)
        pickle.dump({
            'detections': {
                k: {
                    'image_points': torch.from_numpy(v).to(chessboard),
                    'object_points': chessboard
                }
                for k, v in results.items()
                if v is not None
            }
        }, f)

    num_plots = len(results)
    cols = int(math.ceil(math.sqrt(num_plots)))
    rows = int(math.ceil(num_plots / cols))
    fig, ax = plt.subplots(rows, cols, squeeze=False, figsize=(12, 12))
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99, 0.0, 0.0)
    ax = ax.flatten()
    for a in ax:
        a.axis('off')
    for a, (p, corners) in zip(ax, sorted(results.items())):
        img = cv.imread(str(p))
        H, W = img.shape[:2]
        f = 512 / max(H, W)
        img = cv.cvtColor(cv.resize(img, None, fx=f, fy=f), cv.COLOR_BGR2RGB)
        a.imshow(img, extent=[-0.5, W - 0.5, H - 0.5, -0.5])
        if corners is not None:
            lines = np.concatenate((
                corners[:-1, None], corners[1:, None]),
                axis=1
            )
            a.add_collection(LineCollection(
                lines,
                colors=plt.get_cmap('plasma')(
                    np.linspace(0, 1, len(lines))
                ),
                zorder=1
            ))
            a.scatter(corners[:, 0], corners[:, 1], marker='+', color='b',
                      zorder=2)

        a.text(0.5, 0.75, p.stem, color='b', horizontalalignment='center',
               transform=a.transAxes)
    fig.savefig('corners.pdf', dpi=300)
    plt.show()
    plt.close(fig)
