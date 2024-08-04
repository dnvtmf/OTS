from pathlib import Path
import argparse
import open3d as o3d


def options():
    parser = argparse.ArgumentParser('Simplify or Subdivison Mesh')
    parser.add_argument('mesh', help='The path of mesh')
    parser.add_argument('-n', '--num-triangles', default=500_000, type=int, help='The number of target triangles')
    parser.add_argument('-e', '--max-error', default=0.1, help='To control ')
    parser.add_argument('-o', '--output', default='', help='The output path')
    return parser.parse_args()


def main():
    args = options()
    mesh = o3d.io.read_triangle_mesh(args.mesh)  # type: o3d.geometry.TriangleMesh
    num_old = len(mesh.triangles)
    if len(mesh.triangles) > args.num_triangles:
        mesh = mesh.simplify_quadric_decimation(args.num_triangles, args.max_error)
        print(f"simplify mesh from {num_old} to {len(mesh.triangles)} triangles")
    else:
        print(f"There are {num_old} triangles, do not need to simplify")
    output_path = Path(args.output) if args.output else Path(args.mesh)
    assert output_path.suffix == '.obj'
    o3d.io.write_triangle_mesh(output_path.as_posix(), mesh)


if __name__ == '__main__':
    main()
