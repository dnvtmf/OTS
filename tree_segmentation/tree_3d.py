import argparse
import math
from pathlib import Path
from typing import Dict, List, Union, Optional

import numpy as np
import nvdiffrast.torch as dr
import torch
from torch import Tensor
import xatlas

from tree_segmentation.extension import Mesh, utils
from tree_segmentation.extension import ops_3d
from semantic_sam import SemanticSAM, semantic_sam_l
import tree_segmentation as tree_segmentation
from segment_anything.build_sam import Sam, build_sam
from tree_segmentation import MaskData, TreeData, TreePredictor, Tree3D
from tree_segmentation.render import render_mesh
from tree_segmentation.util import get_hash_name


class TreeSegment:

    def __init__(self, args, model: Optional[Union[Sam, SemanticSAM]] = None) -> None:
        self.cfg = args
        self.device = torch.device('cuda:0')
        self.glctx = dr.RasterizeCudaContext()
        if self.cfg is None:
            self.mesh_path = Path('~/data/meshes/tang_table/mesh_0_clean.obj').expanduser()
        else:
            self.mesh_path = Path(self.cfg.mesh).expanduser()
        self.cache_root = Path('./results')
        self.cache_root.mkdir(exist_ok=True)
        self.cache_dir = self.cache_root

        self._model_type = 'SAM'
        self._model: Optional[Union[Sam, SemanticSAM]] = model
        self._mesh: Optional[Mesh] = None
        self._predictor: Optional[TreePredictor] = None

        self._tree_2d: Optional[TreeData] = None
        self._2d_aux_data: Optional[Dict[Union[int, str], Tensor]] = None
        self._image: Optional[np.ndarray] = None
        self._points: Optional[np.ndarray] = None
        self.tri_id: Optional[Tensor] = None
        self.Tw2v: Optional[Tensor] = None
        self._mask_data: Optional[MaskData] = None  # The results of one sample
        self._2d_levels: List[Tensor] = []
        self._2d_mask = None

        self._tree_3d: Optional[Tree3D] = None
        self._3d_levels: List[Tensor] = []
        self._3d_mask = None
        self._3d_aux_data: Optional[Dict[Union[int, str], Tensor]] = None

    def get_value(self, name: str, default=None):
        return getattr(self.cfg, name, default)

    @property
    def model(self):
        if self._model is None:
            self.load_model()
        return self._model

    def load_model(self):
        print('Loading model...')
        if self._model_type == 'SAM':
            self._model = build_sam(self.get_value('sam_path')).to(self.device)
            # predictor = SamPredictor(sam)
        elif self._model_type == 'Semantic-SAM-L':
            self._model = semantic_sam_l(self.get_value('semantic_sam_l_path')).to(self.device)
        elif self._model_type == 'Semantic-SAM-T':
            self._model = semantic_sam_l(self.get_value('semantic_sam_t_path')).to(self.device)
        print(f'Loaded Model: {self._model_type}')

    @property
    def mesh(self) -> Mesh:
        if self._mesh is None:
            self.load_mesh(self.mesh_path, use_cache=True)
        return self._mesh

    @property
    def predictor(self) -> TreePredictor:
        if self._predictor is None:
            self._predictor = tree_segmentation.TreePredictor(
                self.model,
                points_per_batch=self.get_value('points_per_batch', 64),
                pred_iou_thresh=self.get_value('pred_iou_thresh', 0.88),
                stability_score_thresh=self.get_value('stability_score_thresh', 0.95),
                box_nms_thresh=self.get_value('box_nms_thresh', 0.7),
            )
        if not self._predictor.is_image_set and self._image is not None:
            self._predictor.set_image(np.clip(self._image * 255., 0, 255).astype(np.uint8))
        return self._predictor

    @property
    def image(self) -> np.ndarray:
        if self._image is None:
            self.new_camera_pose()
        return self._image

    @image.setter
    def image(self, img: np.ndarray):
        self._image = img

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points):
        self._points = points

    @property
    def tree2d(self) -> TreeData:
        if self._tree_2d is None:
            self._tree_2d = TreeData(
                in_threshold=self.get_value('in_threshold'),
                in_thres_area=self.get_value('in_area_threshold'),
                union_threshold=self.get_value('union_threshold'),
                min_area=self.get_value('min_area'))
        if self._tree_2d.data is None and self.tri_id is not None:
            background = self.tri_id.eq(0)
            foreground = torch.logical_not(background)
            mask_data = MaskData(
                masks=torch.stack([foreground, background], dim=0),
                iou_preds=torch.ones(2, device=self.tri_id.device) * 2,
                points=torch.zeros((2, 2), dtype=torch.float64),
                stability_score=torch.ones(2, device=self.tri_id.device) * 2,
                boxes=torch.zeros((2, 4), device=self.tri_id.device),
            )
            self._tree_2d.cat(mask_data)
            self._tree_2d.update_tree()
        return self._tree_2d

    @property
    def tree3d(self) -> Tree3D:
        if self._tree_3d is None:
            self._tree_3d = tree_segmentation.Tree3D(self.mesh, device=self.device)
        return self._tree_3d

    @property
    def mask_data(self):
        return self._mask_data

    @mask_data.setter
    def mask_data(self, data: MaskData = None):
        self._mask_data = [] if data is None else data

    @property
    def levels_2d(self):
        return self._2d_levels

    @levels_2d.setter
    def levels_2d(self, levels):
        # max_levels = len(levels)
        # for i in range(1, 10):
        #     dpg.configure_item(f"level{i}", show=i < max_levels)
        self._2d_levels = levels

    @property
    def levels_3d(self):
        return self._3d_levels

    @levels_3d.setter
    def levels_3d(self, levels: list):
        # max_levels = len(levels)
        # for i in range(1, 10):
        #     dpg.configure_item(f"depth{i}", show=i < max_levels)
        self._3d_levels = levels

    @property
    def aux_data_2d(self):
        if self._2d_aux_data is None and self.tri_id is not None:
            self._2d_aux_data = self.tree3d.get_aux_data(self.tri_id)
        return self._2d_aux_data

    @property
    def aux_data_3d(self):
        return self._3d_aux_data

    def load_mesh(self, obj_path, use_cache=False, cache_suffix='.mesh_cache'):
        obj_path = Path(obj_path).expanduser()
        if obj_path.is_dir():
            part_paths = sorted(list(obj_path.glob('*.obj')))
            if len(part_paths) == 0:
                print(f"[GUI] There are no *.obj files in dir {self.mesh}")
                return False
        self.mesh_path = obj_path
        self.cache_dir = self.cache_root.joinpath(get_hash_name(self.mesh_path))
        self.cache_dir.mkdir(exist_ok=True)

        self._tree_3d = None
        self.reset_3d()
        self.reset_2d()

        print(f'[GUI] set cache dir: {self.cache_dir}, use cache: {use_cache}')
        cache_file = self.cache_dir.joinpath(self.mesh_path.stem + cache_suffix)
        if use_cache is True and cache_file.is_file():
            mesh = torch.load(cache_file, map_location=self.device)
            print('[GUI] Load from cache:', cache_file)
        else:
            if self.mesh_path.is_dir():
                part_paths = sorted(list(self.mesh_path.glob('*.obj')))
                print(f"[GUI] Loaded {len(part_paths)} parts from dir '{self.mesh_path}'")
                part_meshes = [Mesh.load(part_path, mtl=True) for part_path in part_paths]
                mesh = Mesh.merge(*part_meshes)
            else:
                mesh = Mesh.load(self.mesh_path)
            mesh.float()
            mesh.int()
            mesh = mesh.cuda().unit_size()
            if mesh.v_tex is not None:
                mesh.compuate_normals_()
                mesh.compute_tangents_()
            print('[GUI] Loaded Mesh from file', self.mesh_path)
            if use_cache:
                torch.save(mesh, cache_file)
                print('[GUI] Save cached mesh:', cache_file)
            print(mesh)
        self._mesh = mesh
        return True

    def reset_2d(self):
        print('[GUI] reset 2D')
        self._tree_2d = None
        self.mask_data = None
        self._2d_mask = None
        self._points = None
        self._2d_aux_data = None
        self.levels_2d = []

    def reset_3d(self):
        print('[GUI] reset 3D')
        self.levels_3d = []
        self._3d_mask = None
        self._2d_aux_data = None
        self._3d_aux_data = None
        if self._tree_3d is not None:
            self._tree_3d.reset()

    def render_mesh(self, Tw2v: Tensor, image_size=1024, light_location=(0, 2., 0.)):
        return render_mesh(
            self.glctx, self.mesh, Tw2v=Tw2v.to(self.device), image_size=image_size, light_location=light_location)
        # mesh: Mesh = self.mesh
        # Tw2v = Tw2v.to(self.device)
        # camera_pos = Tw2v.inverse()[:3, 3]
        # view_direction = ops_3d.normalize(camera_pos)
        # lights = ops_3d.PointLight(
        #     ambient_color=utils.n_tuple(0.5, 3),
        #     diffuse_color=utils.n_tuple(1., 3),
        #     specular_color=utils.n_tuple(0.3, 3),
        #     device=self.device
        # )
        # lights.location = camera_pos
        # fovy = self.get_value('fovy', math.radians(60))
        # Tv2c = ops_3d.perspective(fovy=fovy, size=(image_size, image_size), device=self.device)
        # v_pos = ops_3d.xfm(mesh.v_pos.float(), Tv2c @ Tw2v)
        # v_pos = v_pos[None] if v_pos.ndim == 2 else v_pos
        # rast, _ = dr.rasterize(self.glctx, v_pos, mesh.f_pos.int(), (image_size, image_size))
        # points, _ = dr.interpolate(mesh.v_pos[None], rast, mesh.f_pos.int())
        # # mask = rast[..., -1]>0
        # # print(utils.show_shape(points, Tw2c))
        # # z_w = ops_3d.xfm(points, Tw2c[:, None, :, :])
        # # print(utils.show_shape(z_w, rast, mask))
        # # print((rast[..., 2] - z_w[..., 2]/z_w[..., -1])[mask])
        # if mesh.f_tex is not None:
        #     uv, uv_da = dr.interpolate(mesh.v_tex[None], rast, mesh.f_tex.int())
        #     ka = dr.texture(mesh.material['ka'].data[..., :3].contiguous(), uv) if 'ka' in mesh.material else 0
        #     kd = dr.texture(mesh.material['kd'].data[..., :3].contiguous(), uv) if 'kd' in mesh.material else 0
        #     ks = dr.texture(mesh.material['ks'].data, uv) if 'ks' in mesh.material else 0
        #     nrm = ops_3d.compute_shading_normal(mesh, camera_pos, rast, None)
        # else:
        #     nrm = ops_3d.compute_shading_normal_face(mesh, camera_pos[None, None, None], rast, None)
        #     ka, kd, ks = nrm.new_full((3,), 0.2), nrm.new_full((3,), 0.5), nrm.new_full((3,), 0.1)
        # images = ops_3d.Blinn_Phong(nrm, lights(points), view_direction, (ka, kd, ks)).clamp(0, 1)
        # images = dr.antialias(images, rast, v_pos, mesh.f_pos.int())
        # images = torch.where(rast[..., -1:] > 0, images, torch.ones_like(images))
        # return images[0, :, :, :3], rast[0, :, :, -1].int()

    @torch.no_grad()
    def rendering(self, Tw2v, fovy=None, size=(1024, 1024)):
        if not isinstance(Tw2v, Tensor):
            eye = ops_3d.coord_spherical_to(*Tw2v).to(self.device)
            if abs(eye[0]) < 1e-6 and abs(eye([2])) < 1e-6:
                Tw2v = ops_3d.look_at(eye, torch.zeros_like(eye), eye.new_tensor([0, 0, 1]))
            else:
                Tw2v = ops_3d.look_at(eye, torch.zeros_like(eye))
        image, tri_id = self.render_mesh(Tw2v, image_size=size[0])
        image = image.cpu().numpy()

        self._3d_aux_data = self.tree3d.get_aux_data(tri_id)
        self.levels_3d = self.tree3d.get_levels(self._3d_aux_data)
        return image

    def run_tree_seg_2d_stage1(self):
        if not self.mask_data:
            self.reset_2d()
        points = self.tree2d.sample_grid(self.get_value('points_per_side'))
        # filter points in background
        x = np.clip(np.rint(points[:, 0] * self.tri_id.shape[1]), 0, self.tri_id.shape[1] - 1).astype(np.int32)
        y = np.clip(np.rint(points[:, 1] * self.tri_id.shape[0]), 0, self.tri_id.shape[0] - 1).astype(np.int32)
        self.points = points[self.tri_id.cpu().numpy()[y, x] > 0].reshape(-1, 2)
        # process points
        self.mask_data = self.predictor.process_points(self.points)
        # dpg.get_item_callback('level0')()
        self.tree2d.cat(self.mask_data)
        self.tree2d.update_tree()
        self.tree2d.remove_not_in_tree()
        self.levels_2d = self.tree2d.get_levels()

    def run_tree_seg_2d_stage2(self):
        if not self.mask_data:
            self.reset_2d()
        # points, unfilled_mask = self.tree_2d.sample_unfilled(
        #     dpg.get_value('points_per_update'), dpg.get_value('filled_threshold')
        # )
        self.points = self.tree2d.sample_by_counts(self.get_value('points_per_update'))
        if self.points is None:
            print(f'[Tree 2D] Update complete')
            return False
        self.mask_data = self.predictor.process_points(self.points)
        self.tree2d.cat(self.mask_data)
        self.tree2d.update_tree()
        self.tree2d.remove_not_in_tree()
        self.levels_2d = self.tree2d.get_levels()
        return True

    def autorun_tree_seg_2d(self):
        self.run_tree_seg_2d_stage1()
        print('[Tree 2D]: autorun stage1')
        max_steps = self.get_value('max_steps', 100)
        for step in range(max_steps):
            if not self.run_tree_seg_2d_stage2():
                break
            print(f'[Tree 2D]: autorun stage1 {step + 1}/{max_steps}')

    def run_tree_seg_2d_post(self):
        if self._tree_2d is not None:
            self.tree2d.post_process()

    def new_camera_pose(self, *, Tw2v=None):
        if Tw2v is None:
            Tw2v = self.tree3d.proposal_camera_pose(
                radius_range=(self.get_value('radius_min'), self.get_value('radius_max')),
                elev_range=(self.get_value('theta_min'), self.get_value('theta_max')),
                azim_range=(self.get_value('phi_min'), self.get_value('phi_max')),
            )[0]
        image, tri_id = self.render_mesh(Tw2v, light_location=(0, 4., 0.))
        self.Tw2v = Tw2v
        self.tri_id = tri_id
        self._image = image.cpu().numpy()
        self.reset_2d()
        if self._predictor is not None:
            self._predictor.reset_image()

    def merge_to_3d(self, *, save=None):
        if self.mask_data is not None and self.tree2d.data is not None:
            self.tree3d.update(self.tree2d.data, self.aux_data_2d)
            if save is None:
                save = self.get_value('save_tree_data')
            if save:
                num = len(list(self.cache_dir.glob('*.data'))) + 1
                torch.save(
                    {
                        'tree_data': self.tree2d.save(filename=None),
                        'image': self.image,
                        'tri_id': self.tri_id,
                        'Tw2v': self.Tw2v,
                    }, self.cache_dir.joinpath(f'{num:04d}.data'))
                print(f'[GUI] save data, index={num}')

    def run_tree_3d_cycle(self):
        N = self.get_value('N_cycle')
        Tw2v = self.tree3d.proposal_camera_pose_cycle(
            N,
            radius_range=(self.get_value('radius_min'), self.get_value('radius_max')),
            elev_range=(self.get_value('theta_min'), self.get_value('theta_max')),
            azim_range=(self.get_value('phi_min'), self.get_value('phi_max')),
        )
        print(f'[GUI] run_tree_3d_cycle: {0}/{N}')
        for i in range(N):
            self.new_camera_pose(Tw2v=Tw2v[i])
            self.autorun_tree_seg_2d()
            self.merge_to_3d()
            print(f'[GUI] run tree_3d_cycle: {i + 1}/{N}')

    def run_tree_3d_uniform(self):
        N = self.get_value('N_uniform')
        Tw2v = self.tree3d.proposal_camera_pose_uniform(
            N,
            radius_range=(self.get_value('radius_min'), self.get_value('radius_max')),
            elev_range=(self.get_value('theta_min'), self.get_value('theta_max')),
            azim_range=(self.get_value('phi_min'), self.get_value('phi_max')),
        )
        print(f'[GUI] run_tree_3d_uniform: {0}/{N}')
        for i in range(N):
            self.new_camera_pose(Tw2v=Tw2v[i])
            self.autorun_tree_seg_2d()
            self.merge_to_3d()
            print(f'[GUI] run tree_3d_uniform: {i + 1}/{N}')

    def run_tree_3d_grid(self):
        N = self.get_value('N_grid')
        Tw2v = self.tree3d.proposal_camera_pose_spherical_grid(N, radius_range=(2.5, 3.0))
        print(Tw2v.shape)
        for i in range(N):
            self.new_camera_pose(Tw2v=Tw2v[i])
            self.autorun_tree_seg_2d()
            self.merge_to_3d()
            print(f'[GUI] run_tree_3d_grid: {i + 1}/{N}')

    def run_tree_3d_load(self, *index: int):
        filenames = sorted(list(self.cache_dir.glob('*.data')))
        if len(index) > 0:
            filenames = [filenames[i] for i in index]
        for i, filename in enumerate(filenames):
            data = torch.load(filename, map_location=self.device)
            self.tri_id = data['tri_id']
            self._2d_aux_data = self.tree3d.get_aux_data(self.tri_id)
            self._image = data['image']
            self.Tw2v = data['Tw2v']
            self.reset_2d()
            self.tree2d.load(filename=None, **data['tree_data'])
            self.mask_data = None
            self.levels_2d = self.tree2d.get_levels()
            self.merge_to_3d(save=False)

    def show_uv_results(self):
        pass

    def save_tree_3d(self, filename='tree_3d.pth'):
        if self._tree_3d is None:
            print(f"[GUI] save Tree3D Failed: no Tree3D")
            return
        self.tree3d.save(self.cache_dir.joinpath(filename))
        print(f"[GUI] save Tree3D to:", self.cache_dir.joinpath(filename))

    def load_tree_3d(self, filename='tree_3d.pth'):
        if not self.cache_dir.joinpath(filename).exists():
            print(f'[GUI] No such file: {self.cache_dir.joinpath(filename)} to load Tree3D')
            return
        self.tree3d.load(self.cache_dir.joinpath(filename))
        print(f"[GUI] load Tree3D from:", self.cache_dir.joinpath(filename))
        self.tree3d.print_tree()

    def compuate_uv(self, tri_id: Tensor = None, mask: Tensor = None, num_adj=2, image_size=1024):
        if tri_id is not None:
            adj_nodes = torch.zeros(self.mesh.v_pos.shape[0], dtype=torch.bool, device=self.device)  # 相邻顶点
            adj_faces = (tri_id if mask is None else tri_id[mask]).unique() - 1  # 相邻边
            if adj_faces[0] < 0:
                adj_faces = adj_faces[1:]
            adj_nodes[self.mesh.f_pos[adj_faces]] = 1
            for _ in range(num_adj):
                adj_faces = adj_nodes[self.mesh.f_pos].any(dim=-1)
                adj_nodes[self.mesh.f_pos[adj_faces]] = 1
            adj_faces = torch.nonzero(adj_nodes[self.mesh.f_pos].any(dim=-1)).squeeze()
            _, f_pos, v_pos = xatlas.parametrize(self.mesh.v_pos.cpu().numpy(),
                                                 self.mesh.f_pos[adj_faces].cpu().numpy())
        else:
            adj_faces = None
            _, f_pos, v_pos = xatlas.parametrize(self.mesh.v_pos.cpu().numpy(), self.mesh.f_pos.cpu().numpy())
        v_pos = torch.from_numpy(v_pos).to(self.device) * 2 - 1  # [0, 1] --> [-1, 1]
        v_pos = torch.cat([v_pos, torch.zeros_like(v_pos[:, :1]), torch.ones_like(v_pos[:, :1])], dim=-1)
        f_pos = torch.from_numpy(f_pos.astype(np.int32)).to(self.device)
        tri_id_2 = dr.rasterize(self.glctx, v_pos[None], f_pos, (image_size, image_size))[0][0, :, :, -1].int()
        if tri_id is not None:
            adj_faces = torch.constant_pad_nd(adj_faces + 1, (1, 0), 0)
            tri_id_2 = adj_faces[tri_id_2]
        return tri_id_2


def options():
    parser = argparse.ArgumentParser()
    utils.add_path_option(parser, '-m', '--mesh', default='~/data/meshes/lego/lego.obj')
    # sam
    utils.add_path_option(parser, '--sam-path', default='~/models/sam_vit_h_4b8939.pth')
    parser.add_argument('--pred_iou_thresh', default=0.88, type=float)
    parser.add_argument('--stability_score_thresh', default=0.95, type=float)
    parser.add_argument('--box_nms_thresh', default=0.7, type=float)
    parser.add_argument('--points_per_batch', default=64, type=int)
    # 2D
    parser.add_argument('--max-steps', default=100, type=int)
    parser.add_argument('--points_per_update', default=256, type=int)
    parser.add_argument('--points_per_side', default=32, type=int)
    parser.add_argument('--in_threshold', default=0.9, type=float)
    parser.add_argument('--in_area_threshold', default=50, type=float)
    parser.add_argument('--union_threshold', default=0.1, type=float)
    parser.add_argument('--min_area', default=100, type=float)
    utils.add_bool_option(parser, '--save_tree_data', default=True)
    # 3D
    parser.add_argument('-Nc', '--N-cycle', default=100, type=int)
    parser.add_argument('-Ng', '--N-grid', default=100, type=int)
    parser.add_argument('-Nu', '--N-uniform', default=100, type=int)
    parser.add_argument('--radius_min', default=2.0, type=float)
    parser.add_argument('--radius_max', default=2.0, type=float)
    parser.add_argument('--theta-min', default=0, type=float)
    parser.add_argument('--theta-max', default=180, type=float)
    parser.add_argument('--phi-min', default=0, type=float)
    parser.add_argument('--phi-max', default=360., type=float)
    parser.add_argument('--fovy', default=60., type=float)
    args = parser.parse_args()
    print(args)
    args.fovy = math.radians(args.fovy)
    return args


def main():
    args = options()
    segment = TreeSegment(args)
    segment.run_tree_3d_cycle()
    segment.run_tree_3d_uniform()
    segment.run_tree_3d_load()


# 概率图模型
## 理论框架
## 图神经网络
## Tree3D
