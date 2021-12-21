import logging
import pickle

from .main import main

logging.basicConfig(level=logging.INFO)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description='Calibrate omnidirectional cameras')
    parser.add_argument('corners', type=str, help='Pickle file with corners')
    parser.add_argument('-d', '--degree', type=int,
                        default=4, help='Polynom degree')
    parser.add_argument('-p', '--principal-point', type=float, nargs=2,
                        help='Principal point relative to image center (if not given image center'
                        ' is assumed)')
    parser.add_argument('-t', '--threshold', type=float, default=10.,
                        help='Reprojection error threshold in pixel for'
                        ' initial solution')
    parser.add_argument('-c', '--count', type=int, default=3,
                        help='Minimal number of images for initial solution')
    args = parser.parse_args()

    with open(args.corners, 'rb') as f:
        data = pickle.load(f)

    try:
        main(
            data['detections'],
            args.degree,
            args.threshold,
            args.count,
            args.principal_point
        )
    except Exception as e:
        import sys
        sys.stderr.write(str(e))
        raise e


if __name__ == '__main__':
    parse_args()
