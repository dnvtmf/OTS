"""
    This script compute transformation-matrix for aligning shapenetv1 mesh to partnet data
        Usage: python compute_transmat_shapenetv1_to_partnet.py [anno_id] [version_id (not used in this script)] [category] [shapenet-model-id]

    Method:
        1) align orientation (for most cases, partnet and shapenetv1 is subject to the same rotation across all models)
            -- for very rare cases, this will not work since shapenet orientation is actively being fixed, 
               so that some (but very few) partnet data is using newer version of orientation annotations than shapenet-v1
        2) normalize partnet mesh using the shapenetv1 criterion (unit-diagonal box)
        3) for the rare cases mentioned in 1), we compute chamfer-distance between the aligned shapenet and the partnet and cut a threshold to detect failed alignment

    Please contact Kaichun Mo for any questions for using this script.
"""

import os
import sys
import numpy as np
import trimesh
from scipy.spatial.distance import cdist

anno_id = sys.argv[1]
cat_name = sys.argv[3]
model_id = sys.argv[4]

shapenet_cat2ids = {
    'Chair': ['03001627'],
    'Table': ['04379243'],
    'Bed': ['02818832'],
    'StorageFurniture': ['02933112', '03337140', '02871439'],
}

shapenet_dir = '[path-to-shapenet-directory]'
partnet_dir = os.path.join('[path-to-partnet-directory]', anno_id)
input_objs_dir = os.path.join(partnet_dir, 'objs')

out_dir = 'output_transmat_shapenetv1_to_partnet'


def load_obj(fn):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []
    faces = []
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))

    return np.vstack(vertices), np.vstack(faces)


vs = []
fs = []
vid = 0
for item in os.listdir(input_objs_dir):
    if item.endswith('.obj'):
        cur_vs, cur_fs = load_obj(os.path.join(input_objs_dir, item))
        vs.append(cur_vs)
        fs.append(cur_fs + vid)
        vid += cur_vs.shape[0]

v_arr = np.concatenate(vs, axis=0)
v_arr_ori = np.array(v_arr, dtype=np.float32)
f_arr = np.concatenate(fs, axis=0)
tmp = np.array(v_arr[:, 0], dtype=np.float32)
v_arr[:, 0] = v_arr[:, 2]
v_arr[:, 2] = -tmp

x_min = np.min(v_arr[:, 0])
x_max = np.max(v_arr[:, 0])
x_center = (x_min + x_max) / 2
x_len = x_max - x_min
y_min = np.min(v_arr[:, 1])
y_max = np.max(v_arr[:, 1])
y_center = (y_min + y_max) / 2
y_len = y_max - y_min
z_min = np.min(v_arr[:, 2])
z_max = np.max(v_arr[:, 2])
z_center = (z_min + z_max) / 2
z_len = z_max - z_min
scale = np.sqrt(x_len**2 + y_len**2 + z_len**2)

trans = np.array([[0, 0, 1.0/scale, -x_center/scale], \
                  [0, 1.0/scale, 0, -y_center/scale], \
                  [-1/scale, 0, 0, -z_center/scale], \
                  [0, 0, 0, 1]], dtype=np.float32)
trans = np.linalg.inv(trans)
with open(os.path.join(out_dir, anno_id + '.npy'), 'wb') as fout:
    np.save(fout, trans)

# test
for synid in shapenet_cat2ids[cat_name]:
    cur_shapenet_dir = os.path.join(shapenet_dir, synid, model_id)
    if os.path.exists(cur_shapenet_dir):
        tmp_mesh = trimesh.load(os.path.join(cur_shapenet_dir, 'model.obj'))
        break

if isinstance(tmp_mesh, trimesh.Scene):
    shapenetv1_mesh = trimesh.util.concatenate(
        tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces) for g in tmp_mesh.geometry.values()))
elif isinstance(tmp_mesh, trimesh.Trimesh):
    shapenetv1_mesh = trimesh.Trimesh(vertices=tmp_mesh.vertices, faces=tmp_mesh.faces)
else:
    raise ValueError('ERROR: failed to correctly load shapenet mesh!')

shapenetv1_vertices = np.array(shapenetv1_mesh.vertices, dtype=np.float32)
shapenetv1_vertices = np.concatenate(
    [shapenetv1_vertices, np.ones((shapenetv1_vertices.shape[0], 1), dtype=np.float32)], axis=1)
shapenetv1_vertices = shapenetv1_vertices @ (trans.T)
shapenetv1_vertices = shapenetv1_vertices[:, :3]

shapenetv1_mesh = trimesh.Trimesh(vertices=shapenetv1_vertices, faces=shapenetv1_mesh.faces)
#with open(os.path.join(out_dir, anno_id+'-aligned.obj'), 'w') as f:
#    f.write(trimesh.exchange.obj.export_obj(shapenetv1_mesh, write_texture=False, include_normals=False, include_color=False, include_texture=False))
shapenetv1_pts = trimesh.sample.sample_surface(shapenetv1_mesh, 2000)[0]

partnet_mesh = trimesh.Trimesh(vertices=v_arr_ori, faces=f_arr - 1)
partnet_pts = trimesh.sample.sample_surface(partnet_mesh, 2000)[0]

dist_mat = cdist(shapenetv1_pts, partnet_pts)
chamfer_dist = dist_mat.min(0).mean() + dist_mat.min(1).mean()
with open(os.path.join(out_dir, anno_id + '.test'), 'w') as fout:
    fout.write('%f\n' % chamfer_dist)
# [IMPORTANT] in practice, the alignment fails if cd > 0.1 or the script fails due to other reasons
