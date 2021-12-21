'''
Detect chessboard marker in images and save corner points in file
'''
from pathlib import Path

from .detect import detect


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Detect chessboard corners in omnidirectional camera'
        ' images')
    parser.add_argument('image_path', type=str, help='Image path')
    parser.add_argument('-c', '--chessboard', type=float, nargs=3,
                        default=(5, 8, 10.),
                        help='Chessboard pattern (rows, cols, square size)')
    parser.add_argument('-m', '--max-dim', type=int,
                        help='Downscale image to maximal dimension to speed'
                             ' upcorner detection (subpixel seach on full'
                             ' resoultion)')
    parser.add_argument('-t', '--threads', type=int, default=2,
                        help='Number of threads')
    args = parser.parse_args()
    detect(
        Path(args.image_path),
        args.threads,
        chessboard_shape=tuple(int(x) for x in args.chessboard[:2]),
        chessboard_size=args.chessboard[2],
        max_dim=args.max_dim
    )


if __name__ == '__main__':
    main()
