import argparse
import open3d as o3d
from detectron2.data import detection_utils as utils


def options():
    parser = argparse.ArgumentParser('Simplify or Subdivison Mesh')
    parser.add_argument('mesh', help='The path of mesh')
    parser.add_argument('-n', '--num-triangles', default=100_000, type=int, help='The number of target triangles')
    parser.add_argument('-e', '--max-error', default=0.1, help='To control ')
    return parser.parse_args()


def main():
    args = options()
    mesh = o3d.io.load


if __name__ == '__main__':
    main()