import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import struct
import collections
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.sparse
import torch_geometric.nn as pyg_nn
import yaml
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torch import nn
from torch_scatter import scatter
from tqdm import tqdm
import seaborn as sns

from tree_segmentation.extension.gaussian_splatting import render, topk_weights, GaussianRasterizationSettings
from tree_segmentation.extension import utils, ops_3d, DictMeter
from tree_segmentation.extension.utils.gui.viewer_3D import Viewer3D
from tree_segmentation.tree_2d_segmentation import Tree2D
from tree_segmentation.tree_3d_segmentation import AutoEncoder
from tree_segmentation.tree_structure import TreeStructure
from tree_segmentation.loss import get_mask
from tree_segmentation.gaussian_splatting import GaussianSplatting
import tree_segmentation.extension.utils.io.colmap as colmap_util
from evaluation.util import get_predictor, predictor_options


# from NeRF.networks.gaussian_splatting import GaussianSplatting


def load_model() -> GaussianSplatting:
    model_path = '/home/wan/Projects/NeRF/gaussian-splatting/output/7f156431-5/point_cloud/iteration_30000/point_cloud.ply'
    model = GaussianSplatting()
    model.load_ply(model_path)
    model.active_sh_degree = model.max_sh_degree
    model.cuda()
    return model


def vis_model(model: GaussianSplatting):
    import dearpygui.dearpygui as dpg
    @torch.no_grad()
    def rendering(Tw2v, fovy, size):
        Tw2v = Tw2v.cuda()
        Tw2v = ops_3d.convert_coord_system(Tw2v, 'opengl', 'colmap')
        fovx = ops_3d.fovx_to_fovy(fovy, size[1] / size[0])
        Tv2c = ops_3d.perspective_v2(fovy, size=size).cuda()
        # print(Tv2c_2)
        # exit()
        Tw2c = Tv2c @ Tw2v
        Tv2w = torch.inverse(Tw2v)
        tanfovx = math.tan(0.5 * fovx)
        tanfovy = math.tan(0.5 * fovy)
        bg_color = [1, 1, 1]  # if dataset.background == 'white' else [0, 0, 0]
        bg_color = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        t = torch.tensor([dpg.get_value('time')]).cuda()
        raster_settings = GaussianRasterizationSettings(
            image_height=size[1],
            image_width=size[0],
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            scale_modifier=1.0,
            viewmatrix=Tw2v.T,
            projmatrix=Tw2c.T,
            sh_degree=model.max_sh_degree,
            campos=Tv2w[:3, 3],
            prefiltered=False,
            debug=False
        )
        return render(**model(t=t, campos=Tv2w[:3, 3]), raster_settings=raster_settings)['images']

    dpg.create_context()
    dpg.create_viewport(title='Gaussian Splatting')
    is_vary_time = False
    with dpg.window(tag='Primary Window'):
        img = Viewer3D(rendering, size=(800, 800), no_resize=False, no_move=True)
        with dpg.window(tag='control', width=200):
            dpg.add_text(tag='fps')
            with dpg.group():
                dpg.add_text('fovy')
                dpg.add_slider_float(
                    min_value=15.,
                    max_value=180.,
                    default_value=math.degrees(img.fovy),
                    callback=lambda *args: img.set_fovy(dpg.get_value('set_fovy')),
                    tag='set_fovy'
                )
            with dpg.group():
                dpg.add_text('camera pos:')
                dpg.add_input_float(tag='eye_x')
                dpg.add_input_float(tag='eye_y')
                dpg.add_input_float(tag='eye_z')

                def change_eye(*args):
                    print('change camera position', args)
                    img.eye = img.eye.new_tensor([dpg.get_value(item) for item in ['eye_x', 'eye_y', 'eye_z']])
                    img.need_update = True

                dpg.add_button(label='change', callback=change_eye)
            with dpg.group(horizontal=True):
                dpg.add_slider_float(label='t', tag='time', max_value=1.0, callback=img.set_need_update)

                def vary_time():
                    nonlocal is_vary_time
                    is_vary_time = not is_vary_time

                dpg.add_button(label='A', callback=vary_time)

    with dpg.handler_registry():
        dpg.add_mouse_drag_handler(callback=img.callback_mouse_drag)
        dpg.add_mouse_wheel_handler(callback=img.callback_mouse_wheel)
        dpg.add_mouse_release_handler(callback=img.callback_mouse_release)

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window('Primary Window', True)
    # dpg.start_dearpygui()
    last_size = None
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
        if is_vary_time:
            t = dpg.get_value('time')
            t = t + 0.01
            if t > 1:
                t = 0.
            dpg.set_value('time', t)
            img.need_update = True
        if img.need_update:
            dpg.set_value('eye_x', img.eye[0].item())
            dpg.set_value('eye_y', img.eye[1].item())
            dpg.set_value('eye_z', img.eye[2].item())
        img.update()
        now_size = dpg.get_item_width(img._win_id), dpg.get_item_height(img._win_id)
        if last_size != now_size:
            dpg.configure_item('control', pos=(dpg.get_item_width(img._win_id), 0))
            dpg.set_viewport_width(dpg.get_item_width(img._win_id) + dpg.get_item_width('control'))
            dpg.set_viewport_height(dpg.get_item_height(img._win_id))
            last_size = now_size
        dpg.set_value('fps', f"FPS: {dpg.get_frame_rate()}")
    dpg.destroy_context()


class GSTreeSegmentation(TreeStructure):
    _save_list = ['masks', 'scores', 'parent', 'last', 'next', 'first', 'cnt', 'face_mask']

    def __init__(self, num_points, device=None, in_threshold=0.8, union_threshold=0.1, debug=False):
        self.num_points = num_points
        super().__init__(device=device, verbose=0)
        self.score_threshold = 0.5
        self.in_threshold = in_threshold
        self.union_threshold = union_threshold
        self.masks: Optional[Tensor] = None
        self.masks_area: Optional[Tensor] = None
        self.scores: Optional[Tensor] = None
        self.debug = debug
        print(f"[Tree3D] There are {self.num_points} points")

    def reset(self):
        super().reset()
        ## data
        self.tree2ds = []  # type: List[Tensor]
        # self.view_infos = []  # type: List[Tuple[Tensor, Tensor]]
        self.range_view = []  # type: List[Tuple[int, int]]
        self.range_level = []  # type: List[Tuple[int, int]]
        self.masks_view = None  # type: Optional[Tensor]
        self.indices_view = None  # type: Optional[Tensor]
        # self.faces_masks = []  # type: List[Tensor]
        self.range_2d = []  # type: List[Tuple[int, int]]
        self.masks_2d = None  # type: Optional[Tensor]
        self._masks_2d_sp = None  # type: Optional[Tensor]
        self.indices_2d = None  # type: Optional[Tensor]
        self._masks_2d_packed = False

        self.M = 0  # the total number of masks for all views
        self.V = 0  # the total number of views
        ## results
        self.masks = None
        self.scores = None
        torch.cuda.empty_cache()

    def insert(self, i, now=0):
        area_i = self.masks_area[i - 1]
        mask_i = self.masks[i - 1, 1:]
        # bbox_i = self['bbox'][i - 1]
        nodes_in_i = []
        nodes_union = []
        for j in self.get_children(now):
            mask_j = self.masks[j - 1, 1:]
            area_j = self.masks_area[j - 1]
            inter = torch.sum(mask_i * mask_j)
            union = area_i + area_j - inter
            if inter / area_i >= self.in_threshold:  # i in j
                if inter / area_j >= self.in_threshold:  # i == j
                    # print(f"{i} same {j}: area:{area_i}/{area_j},inter={inter}/{inter/area_i:.2%}/{inter/area_j:.2%}")
                    # print(f'inter area: {area_i - inter}/{area_j-inter} < {self.in_thres_area}')
                    if self.scores[i - 1] > self.scores[j - 1]:
                        self.node_replace(i, j)
                    else:
                        if self.verbose > 1:
                            print(f'[Tree3D] {i} is same with {j}, skip')
                    return
                if self.verbose > 1:
                    print(f'[Tree3D] {i} in {j}, {inter.item() / area_i.item():.2%}')
                if self.first[j] < 0:
                    self.node_insert(i, j)
                else:
                    self.insert(i, j)
                return
            elif inter / area_j >= self.in_threshold:  # j in i
                nodes_in_i.append(j)
            elif inter / union > self.union_threshold:
                nodes_union.append((j, (inter / union).item()))
            else:  # no intersect
                pass
        if len(nodes_union) > 0:
            print(f"[Tree3D] {i} union with {nodes_union}")
            return
        # assert len(nodes_union) == 0, f"{i} union with {nodes_union}"
        if self.verbose > 1:
            print(f"[Tree3D] {i} in {now} before {self.first[now].item()}", nodes_in_i)
        self.node_insert(i, now)
        if len(nodes_in_i) > 0:
            for j in nodes_in_i:
                self.node_move(j, i)
                if self.verbose > 2:
                    print(f"[Tree3D] move {j} from {now} to {i}")
        return

    def update_tree(self, threshold=0.5):
        while self.cnt < len(self.masks):
            self.insert(self.node_new())
        # self.print_tree()

    @torch.no_grad()
    def load_2d_results(self, save_dir: Path):
        tree_2d_png_files = sorted(list(save_dir.glob('*.png')))
        tree_2d_results = []
        for png_file in tree_2d_png_files:
            parts = png_file.stem.split('_')
            assert len(parts) == 4 and parts[2] == 'level'
            img_idx = int(parts[1])
            level = int(parts[3])
            # mask = utils.load_image(png_file)
            mask = np.array(Image.open(png_file))
            tree_2d_results.extend([[] for _ in range((img_idx + 1 - len(tree_2d_results)))])
            assert len(tree_2d_results) > img_idx
            tree_2d_results[img_idx].append(mask)
            assert level == len(tree_2d_results[img_idx])
        return tree_2d_results

    def get_mask_2d_weights(self, index: Tensor, weight: Tensor, weight_max: Tensor, mask=None, threshold=0.5):
        if mask is not None:
            index = index.clone()
            weight = weight.clone()
            mask = torch.logical_not(mask).expand_as(weight)
            index[mask] = -1
            weight[mask] = 0
        mask = index > 0
        temp = torch.zeros(self.num_points, device=index.device)
        temp = torch.scatter_reduce(temp, 0, index[mask].long(), weight[mask], 'sum')
        # print((temp > 0).sum(), temp.aminmax(), (weight > 0).sum(), weight.aminmax())
        # plt.figure(dpi=200)
        # plt.subplot(121)
        # plt.hist(temp[temp > 0].cpu().numpy(), bins=100)
        # plt.subplot(122)
        # plt.hist(weight[weight > 0].view(-1).cpu().numpy(), bins=100)
        # plt.show()
        return temp.gt(0.01).float()
        # temp = temp / weight_max.clamp_min(1e-5)
        # return (temp >= threshold).float()

    @torch.no_grad()
    def _get_node_relationship(self, t: TreeStructure):
        M = t.cnt
        v_p = torch.zeros((M, M), dtype=torch.bool, device=self.device)  # all parents
        v_c = torch.zeros((M, M), dtype=torch.bool, device=self.device)  # all sub-tree nodes

        def _query(p=0):
            for c in t.get_children(p):
                if p != 0:
                    v_p[c - 1] |= v_p[p - 1]
                    v_p[c - 1, p - 1] = True
                _query(c)
                if p != 0:
                    v_c[p - 1] |= v_c[c - 1]
            if p != 0:
                v_c[p - 1, p - 1] = True
                v_c[p - 1] |= v_p[p - 1]

        _query()
        v_c = torch.logical_not(v_c)  # all not intersect nodes
        return torch.stack([v_p, v_c])

    def init_from_2D_reults(self, tree_seg_2d, indices: Tensor, weights: Tensor, weights_max: Tensor, pack=False):
        torch.cuda.empty_cache()
        self._masks_2d_packed = pack
        masks_2d = []
        indices_2d = []
        self.range_2d = []
        indices_view = []
        self.range_view = []
        self.range_level = []
        # self.view_infos = []
        self.tree2ds = []
        self.masks_view = torch.zeros((len(tree_seg_2d), self.num_points), dtype=torch.bool, device=self.device)
        self.V = 0
        self.M = 0
        self.Lmax = 0
        # temp = torch.zeros(self.num_points, dtype=torch.float, device=self.device)
        mask_2d = torch.zeros(self.num_points, dtype=torch.int, device=self.device)
        for vid in tqdm(range(len(tree_seg_2d))):
            if self.debug and vid == 3:
                break
            view_weight = self.get_mask_2d_weights(indices[vid], weights[vid], weights_max)
            # print(view_weight.shape)

            num_masks_start = self.M
            num_levels = 0
            for level, masks in enumerate(tree_seg_2d[vid]):
                # if level == 0:  # skip root
                #     continue
                mask_2d.zero_()
                nodes = np.unique(masks)
                if nodes[0] <= 0:
                    nodes = nodes[1:]
                for i in nodes:
                    mask = torch.from_numpy(masks == i)
                    temp = self.get_mask_2d_weights(indices[vid], weights[vid], weights_max, mask)  # / view_weight
                    self.M += 1
                    if pack:
                        mask_2d[temp >= 0.5] = self.M  # TODO: two mask may overlap due to opacity
                        indices_2d.append(len(masks_2d))
                    else:
                        masks_2d.append(temp.clone())
                    indices_view.append(self.V)
                if pack:
                    self.range_2d.append((self.M - len(nodes), self.M))
                    masks_2d.append(mask_2d.clone())
                num_levels += 1

            num_masks_v = self.M - num_masks_start
            if num_masks_v == 0:
                print(f"[Tree3D] load 2d results: view {vid} no vaild masks")
                continue

            tree2d = Tree2D(device=self.device)
            for masks in tree_seg_2d[vid]:
                masks = torch.from_numpy(masks.copy()).cuda()
                mask_idx = torch.unique(masks)
                if mask_idx[0] == 0:
                    mask_idx = mask_idx[1:]
                masks = masks[None, :, :].eq(mask_idx[:, None, None])
                tree2d.insert_batch(masks)
            # if vid == 0:
            #     tree2d.print_tree()
            self.tree2ds.append(self._get_node_relationship(tree2d))
            # self.view_infos.append((v_faces, v_cnts))
            self.masks_view[self.V, view_weight > 0] = 1  # cnts.float()
            self.range_view.append((num_masks_start, self.M))
            if pack:
                self.range_level.append((len(masks_2d) - num_levels, len(masks_2d)))
            self.V += 1
            self.Lmax = max(self.Lmax, num_masks_v)
        if self.V == 0:
            return False
        if pack:
            self.indices_2d = torch.tensor(indices_2d, dtype=torch.int32, device=self.device)
        else:
            self.indices_2d = None
            self.range_2d = None
            self.range_level = None
        self.masks_2d = torch.stack(masks_2d, dim=0).to(self.device)
        print(self.masks_2d.shape)
        self.indices_view = torch.tensor(indices_view, dtype=torch.int32, device=self.device)
        self.masks_view = self.masks_view[:self.V]
        # print(self.range_view, utils.show_shape(self.face_masks))
        # if pack:
        #     indices = torch.nonzero(self.masks_2d)
        #     self._masks_2d_sp = torch.sparse.FloatTensor(  # noqa
        #         torch.stack([self.masks_2d[indices[:, 0], indices[:, 1]] - 1, indices[:, 1]]),
        #         torch.ones(indices.shape[0], device=indices.device),
        #         [self.M, self.num_points],
        #     )
        # else:
        #     self._masks_2d_sp = None
        self.face_mask = F.pad(self.masks_view.any(0), (1, 0))

        print(f'[Tree3D] view_masks: {utils.show_shape(self.masks_view)}')
        print(f"[Tree3D] loaded {self.V} views, {self.M} masks, max_num: {self.Lmax}")
        print('[Tree3D] GPU:', utils.get_GPU_memory())
        return True

    def build_view_graph(self, threshold=0.5, num_nearest=5):
        if self.verbose > 0:
            print(f"[Tree3D] start build view graph")
        assert self.masks_view is not None
        area = torch.sum(self.masks_view.float(), -1)
        # print(area.shape)
        A = F.linear(self.masks_view.float(), self.masks_view.float())
        # A = A / (area[:, None] + area[None, :] - A).clamp_min(1e-7)
        A = A / area[:, None]
        indices = torch.topk(A, min(self.V, num_nearest + 1), dim=0)[1]
        # print(utils.show_shape(indices))
        A = A.ge(threshold)
        A[torch.arange(self.V), indices] = 1
        # print(A.sum(dim=1))
        # self.view_G = A
        return A

    def build_graph(self, view_graph: Tensor = None):
        if self.verbose > 0:
            print(f"[Tree3D] start build graph")
        # view_graph = self.build_view_graph()
        if view_graph is None:
            view_graph = torch.ones((self.V, self.V), device=self.device, dtype=torch.bool)
        if self._masks_2d_packed:
            M = self.M + 1
            area = torch.zeros(M, device=self.device)
            A = torch.zeros((M, M), device=self.device, dtype=torch.float)
            for i in range(len(self.masks_2d)):
                xi, yi = self.range_2d[i][0] + 1, self.range_2d[i][1] + 1
                vi = self.indices_view[xi - 1]
                for j in range(len(self.masks_2d)):
                    xj, yj = self.range_2d[j][0] + 1, self.range_2d[j][1] + 1
                    vj = self.indices_view[xj - 1]
                    if vi >= vj or not view_graph[vi, vj]:
                        continue
                    view_mask = (self.masks_view[vi] & self.masks_view[vj])
                    if view_mask.sum() == 0:
                        A[xi:yi, xj:yj] = 0
                        A[xj:yj, xi:yi] = 0
                        continue
                    area_now = torch.ones_like(view_mask, dtype=torch.float32)
                    idx_i = self.masks_2d[i][view_mask].long()
                    idx_j = self.masks_2d[j][view_mask].long()
                    area[xi:yi] = 0
                    area[xj:yj] = 0
                    area.scatter_reduce_(0, idx_i, area_now, reduce='sum')
                    area.scatter_reduce_(0, idx_j, area_now, reduce='sum')
                    A.view(-1).scatter_reduce_(0, idx_i * M + idx_j, area_now, reduce='sum')
                    inter = A[xi:yi, xj:yj]
                    iou = inter / (area[xi:yi, None] + area[None, xj:yj] - inter).clamp(1e-7)
                    A[xi:yi, xj:yj] = iou
                    A[xj:yj, xi:yi] = iou.T
            A = A[1:, 1:].contiguous()
        else:
            A = torch.zeros((self.M, self.M), device=self.device, dtype=torch.float)
            for i in range(self.V):
                si, ei = self.range_view[i]
                masks_i = self.masks_2d[si:ei].to(self.device)
                for j in range(self.V):
                    if i >= j or not view_graph[i, j]:
                        continue
                    view_mask = (self.masks_view[i] * self.masks_view[j]).float()
                    sj, ej = self.range_view[j]
                    masks_i_ = masks_i * view_mask
                    masks_j = self.masks_2d[sj:ej].to(self.device) * view_mask
                    inter = F.linear(masks_i_, masks_j)
                    area_i = masks_i_.sum(-1)
                    area_j = masks_j.sum(-1)
                    iou = inter / (area_i[:, None] + area_j[None, :] - inter).clamp_min(1e-7)
                    A[si:ei, sj:ej] = iou
                    A[sj:ej, si:ei] = iou.T
        # self.A = A
        torch.cuda.empty_cache()
        return A

    def build_all_graph(self, threshold=0.5, num_nearest=5):
        """build an adjacency matrix between view and view, view and masks, masks and masks"""
        M = self.M
        N = self.M + self.V
        A = torch.zeros((N, N), device=self.device)
        # view-view
        # areas_v = torch.mv(self.masks_view.float(), self.area)[:, None]
        # A[self.M:, self.M:] = F.linear(self.masks_view.float(), self.masks_view * self.area) / areas_v
        A[self.M:, self.M:] = self.build_view_graph(threshold, num_nearest)
        # view-mask
        if self._masks_2d_packed:
            areas_m = torch.zeros((self.M + 1), device=self.device)
            # for i in range(len(self.masks_2d)):
            #     areas_m.scatter_reduce_(0, self.masks_2d[i.long(), self.area, 'sum')
            for i in range(self.M):
                mask = self.masks_2d[self.indices_2d[i]] == (i + 1)
                areas_m[i + 1] = torch.sum(mask)
            temp = torch.zeros((self.M + 1), device=self.device)
            for i in range(self.V):
                masks_2d = self.masks_2d * self.masks_view[i]
                temp.zero_()
                # for j in range(len(masks_2d)):
                #     temp.scatter_reduce_(0, masks_2d[j].long(), self.area, 'sum')
                for j in range(self.M):
                    mask = masks_2d[self.indices_2d[j]] == (j + 1)
                    temp[j + 1] = torch.sum(mask)
                A[:M, M + i] = temp[1:] / areas_m[1:].clamp_min(1e-7)
        else:
            areas_m = torch.sum(self.masks_2d, -1)[:, None].clamp_min(1e-7)
            A[:M, M:] = F.linear(self.masks_2d, self.masks_view.float()) / areas_m
            A[M:, :M] = A[:M, M:].T
        # mask-mask
        A[:M, :M] = self.build_graph()
        return A

    @torch.enable_grad()
    def compress_masks(self, hidden_dims=(256, 256, 256), epochs=3000, batch_size=64, lr=1e-3, include_views=True):
        autoencoder = AutoEncoder(self.num_points, hidden_dims).to(self.device)
        print('AutoEncoder:', autoencoder)
        metric = DictMeter()
        opt = torch.optim.Adam(autoencoder.parameters(), lr=lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, 1e-6)

        autoencoder.train()
        N = self.M + (self.V if include_views else 0)
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(epochs):
            opt.zero_grad(set_to_none=True)
            losses = {}
            # threshold = torch.rand(1, device=device)
            # edges = torch.nonzero(A.gt(threshold))
            # assert len(edges) > 0
            # edges = edges[torch.randint(0, len(edges), (batch_size // 2,))]
            # edges = torch.cat([edges, torch.randint(0, N, (batch_size - edges.shape[0], 2), device=device)], dim=0)
            # gt = faces_masks[edges.view(-1), 1:]
            # print(utils.show_shape(gt), *gt.aminmax())
            indices = torch.randint(0, N, (batch_size,), device=self.device)
            indicesM = indices[indices < self.M] if include_views else indices
            if self._masks_2d_packed:
                masks_gt = (self.masks_2d[self.indices_2d[indicesM]]).eq(indicesM[:, None] + 1)
            else:
                masks_gt = self.masks_2d[indicesM].to(self.device)
            if include_views and len(indices) != len(indicesM):
                indicesV = indices[indices >= self.M] - self.M
                masks_gt = torch.cat([masks_gt, self.masks_view[indicesV].to(masks_gt)], dim=0)
            ## TODO: random project to some view
            # print(utils.show_shape(masks_gt))
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                features_, masks_pred = autoencoder(masks_gt.half())
            # print('features', utils.show_shape(features))

            # view_f = encoder(view_masks[:, 1:])
            # # print('view_f:', utils.show_shape(view_f))
            #
            # mask = view_f[view_indices[edges[:, 0]]] * view_f[view_indices[edges[:, 1]]]
            # masked_feat = features.view(batch_size, 2, -1) * mask[:, None]
            # # print(utils.show_shape(mask, masked_feat))
            # IoU_p = F.cosine_similarity(masked_feat[:, 0], masked_feat[:, 1])
            # IoU_gt = A[edges[:, 0], edges[:, 1]]
            # # print('IoU', utils.show_shape(IoU_gt, IoU_p))
            # loss = F.mse_loss(IoU_p, IoU_gt)
            #
            # recon = decoder(torch.cat([features, view_f, masked_feat.view(-1, dF)], dim=0))
            # mask = view_masks[view_indices[edges[:, 0]]] * view_masks[view_indices[edges[:, 1]]]
            # gt = torch.cat([gt, view_masks[:, 1:], (gt.view(batch_size, 2, -1) * mask[:, None, 1:]).flatten(0, 1)], dim=0)
            # # print('recon mask', utils.show_shape(recon, mask, gt))
            # reco_loss = F.binary_cross_entropy_with_logits(masks_pred, gt)
            # print('recon', utils.show_shape(recon), reco_loss)
            # print('loss:', loss)

            losses['recon'] = F.binary_cross_entropy_with_logits(masks_pred.float(), masks_gt.float())
            metric.update(losses)
            scaler.scale(utils.sum_losses(losses)).backward()
            scaler.step(opt)
            scaler.update()
            lr_scheduler.step()
            if epoch % 100 == 0:
                print(f'[Tree3D] X epoch[{epoch:4d}], loss: {metric.average}, lr={lr_scheduler.get_last_lr()[0]:.3e}')
                metric.reset()
                if self.debug:
                    break
        autoencoder.eval()
        X = []
        with torch.no_grad():
            for indices in torch.arange(self.M, device=self.device).split(batch_size * 2, dim=0):
                if self._masks_2d_packed:
                    masks = (self.masks_2d[self.indices_2d[indices]]).eq(indices[:, None] + 1)
                else:
                    masks = self.masks_2d[indices]
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    X.append(autoencoder(masks.half(), only_encoder=True))
            if include_views:
                for indices in torch.arange(self.V, device=self.device).split(batch_size * 2, dim=0):
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        X.append(autoencoder(self.masks_view[indices].half(), only_encoder=True))
        X = torch.cat(X, dim=0)
        print('[Tree3D] Features of face masks:', utils.show_shape(X))
        # self.X = X
        torch.cuda.empty_cache()
        return X, autoencoder

    def _get_masks(self, P: Tensor, eps=1e-7):
        # P shape: [K, M]
        assert 0 <= self.indices_view.min() and self.indices_view.max() < self.V
        # torch.cuda.synchronize()
        weights = scatter(P, self.indices_view.long(), dim=1, dim_size=self.V, reduce='sum')
        # torch.cuda.synchronize()
        weights = (weights @ self.masks_view.float()).clamp_min(eps)
        assert P.shape[1] == self.M
        if self._masks_2d_packed:
            masks = (P @ self._masks_2d_sp) / weights
        else:
            masks = (P @ self.masks_2d) / weights
        return masks  # shape: [K, F]

    def loss_edge_similarity(self, S: Tensor, A: Tensor):
        S = F.normalize(S, dim=1)  # shape: [K, M]
        sim = S.T @ S  # shape: [M, M]
        return F.mse_loss(sim, A[:self.M, :self.M])

    def loss_2d_tree(self, P: Tensor, masks: Tensor, scores: Tensor, view=-1, eps=1e-7):
        """ The viewed masks like the 2D tree"""
        if view < 0:
            view = torch.randint(0, self.V, (1,)).item()
        masks = masks[:, self.masks_view[view]]  # [K, Fv]
        R = self.tree2ds[view]  # [2, Mv, Mv ] relationship
        s, e = self.range_view[view]
        P = P[:, s:e]  # * scores[:, None]  # [K, Mv]
        # with torch.no_grad():
        #     seen = (masks >= 0.5).any(dim=1)  # [K]
        # P = P[seen]
        # masks = masks[seen]
        # if masks.numel() == 0:
        #     return torch.zeros(1, device=self.device)

        areas = torch.sum(masks, -1)  # [K]
        inter = F.linear(masks, masks)
        In = inter / areas[:, None].clamp_min(eps)  # [K, K]
        IoU = inter / (areas[:, None] + areas[None, :] - inter).clamp_min(eps)  # [K, K]
        notIn = (P.T @ (1 - In) @ P) * R[0]  # [Mv, Mv]
        IoU = (P.T @ IoU @ P) * R[1]  # [Mv, Mv]
        loss = (notIn.sum() + IoU.sum()) / R.sum().clamp_min(eps)
        return loss

    def loss_reg_edge_in_same_view(self, P: Tensor, scores: Tensor):
        """
        Two 2D masks in same view should not map to same 3D mask.

        Args:
            P: shape [K, M]
            scores: shape [K,]

        Returns: Tensor
        """
        mask = (self.indices_view[:, None] == self.indices_view[None, :]) ^ torch.eye(self.M, device=self.device).bool()
        same_view_edges = mask.nonzero(as_tuple=True)
        P_uv = (P[:, same_view_edges[0]] * P[:, same_view_edges[1]]) * scores[:, None]  # shape [K, E]
        return P_uv.sum(dim=0).mean()

    def loss_recon(self, P: Tensor, masks: Tensor, scores: Tensor, A: Tensor, k1=-1, k2=-1, eps=1e-7):
        """ reconstrute A by masks
        A[a, b] = IoU(M[a, k1], M[b, k2]; V[a], V[b])

        Args:
            P: shape [K, M]
            masks: shape [K, F]
            scores: shape [K]
            A: shape [M+V, M+V]

        Returns: Tensor
        """
        if k1 < 0 or k2 < 0:  # random choose two view
            indices = torch.nonzero(A[self.M:, self.M:])
            k1, k2 = indices[torch.randint(0, len(indices), (1,), dtype=torch.long, device=indices.device).item()]
        assert 0 <= k1 < self.V and 0 <= k2 < self.V

        view_mask = self.masks_view[k1] * self.masks_view[k2]
        view_mask[0] = 0
        if view_mask.sum() == 0:  # no intersection
            return torch.zeros(1, device=self.device)
        masks_v = masks[:, view_mask]
        inter = F.linear(masks_v, masks_v)
        area = torch.sum(masks_v, -1)
        IoU = inter / (area[:, None] + area[None, :] - inter).clamp_min(eps)  # tree结点 两两间IoU
        # IoU = 2 * inter_ / (area[:, None] + area[None, :]).clamp_min(eps)  # tree结点 两两间dice score
        IoU = IoU * scores[:, None] * scores[None, :]  # [K, K]

        id_1 = torch.nonzero(self.indices_view.eq(k1))[:, 0]
        id_2 = torch.nonzero(self.indices_view.eq(k2))[:, 0]
        predictions = (P[:, id_1].T @ IoU @ P[:, id_2])
        IoU_gt = A[torch.meshgrid(id_1, id_2, indexing='ij')]
        return F.mse_loss(predictions, IoU_gt)

    def loss_view_mask(self, match_scores: Tensor, scores: Tensor):
        """ Give a view, every 2D mask in this view have a matched 3D mask
        Args:
            match_scores: shape [K, M_v]
            scores: shape [K]
        """
        idx3d, idx2d = linear_sum_assignment((1 - match_scores).detach().cpu().numpy())  # 二分匹配
        idx3d, idx2d = utils.tensor_to(idx3d, idx2d, device=match_scores.device)
        score_vk = match_scores[idx3d, idx2d]
        return 1 - 2 * (score_vk * scores[idx3d]).sum() / (scores[idx3d].sum() + match_scores.size(1))

    def loss_mask_view(self, match_scores: Tensor, scores: Tensor):
        """ Given a view, each 3D masks etheir match a 2D mask or empty or not in results
        Args:
            match_scores: shape [K, M_v]
            scores: shape [K]
        """
        # 二分匹配
        predicton = match_scores.amax(dim=1).clone()
        idx3d, idx2d = linear_sum_assignment((1 - match_scores).detach().cpu().numpy())  # 二分匹配
        idx3d, idx2d = utils.tensor_to(idx3d, idx2d, device=match_scores.device)
        predicton[idx3d] = match_scores[idx3d, idx2d]
        gt = torch.zeros_like(scores, requires_grad=False)
        gt[idx3d] = 1
        # 最大匹配
        # predicton = match_scores.amax(dim=1)  # [K]
        # indices = match_scores.argmax(dim=0)  # [Mv]
        # gt = torch.zeros_like(predicton, requires_grad=False)
        # assert 0 <= indices.min() and indices.max() < len(scores)
        # gt[indices] = 1
        return F.mse_loss(predicton * scores, gt)
        # return (gt - predicton * scores).mean()

    def loss_tree(self, masks: Tensor, scores: Tensor, eps=1e-7):
        """let masks to be a tree"""
        areas = masks.sum(-1)
        inter = F.linear(masks, masks)
        IoU = inter / (areas[:, None] + areas[None, :] - inter).clamp_min(eps)
        In = inter / (areas[:, None]).clamp_min(eps)  # In[i, j] = area(union(i, j)) / area_i
        # 互不相交: inter -> 0 / IoU -> 0
        # 一个完全在另外一个内部  # In[i, j] == area[i] or In[i,j] == area[j]
        # 都不是tree的结点 score[i] -> 0 or score[j] -> 0
        # losses = torch.minimum(IoU, torch.minimum((inter - areas[:, None]).abs(), (inter - areas[None, :]).abs()))
        scores = scores / scores.sum()
        losses = torch.minimum(IoU, 1 - torch.maximum(In, In.T))  # shape: [K, K]
        return torch.sum(losses * scores[None, :])
        # weights = scores[:, None] * scores[None, :]
        # weights = weights / weights.sum()
        # return torch.sum(losses * weights)
        # return losses.mean() * 0.5

    def calc_losses(
        self,
        logits: Tensor,
        node_logits: Tensor,
        view_index: int,
        A: Tensor,
        eps=1e-7,
        progress=1.0,
        timer: utils.TimeWatcher = None,
        weights=None,
    ) -> Dict[str, Tensor]:
        if weights is None:
            weights = dict()
        losses = {}  # type: Dict[str, Tensor]
        if progress < 0 or progress == 1.:
            P = logits.softmax(dim=1).T
        else:
            topP, indices = torch.topk(logits, k=max(1, int(logits.size(1) * (1 - progress))), dim=1)
            topP = topP.softmax(dim=1)
            P = torch.scatter(torch.zeros_like(logits), 1, indices, topP).T

        # masks = self._get_masks(P)  # [K, F]
        masks = get_mask(P, self.masks_2d, self.indices_view, self.masks_view, self._masks_2d_packed, eps=1e-7)
        assert not torch.isnan(masks).any()
        scores = node_logits.float().sigmoid()  # the probability for where a node in the tree
        if timer is not None:
            timer.log('get masks')

        # graph similarity
        if weights.get('es', 0) > 0:
            losses['es'] = self.loss_edge_similarity(P, A)
            if timer is not None:
                timer.log('edge_sim')
        if weights.get('t2d', 1) > 0:
            losses['t2d'] = self.loss_2d_tree(P, masks, scores)
            if timer is not None:
                timer.log('t2d')
        if weights.get('edge', 0) > 0:
            losses['edge'] = self.loss_reg_edge_in_same_view(P, scores)
            if timer is not None:
                timer.log('edge')
        if weights.get('recon', 0) > 0:
            losses['recon'] = self.loss_recon(P, masks, scores, A, eps=eps)
            if timer is not None:
                timer.log('recon')
        # 评估Masks投影到当前view后的masks与当前view检测出的结果之间的差别
        s, e = self.range_view[view_index]
        P = P[:, s:e]
        view_mask = self.masks_view[view_index]  # 当前view的可见部分
        area_3d = torch.mv(masks, view_mask.float())  # shape: [K]
        if self._masks_2d_packed:
            with torch.no_grad():
                masks_2d = []
                for i in range(s, e):
                    assert 0 <= self.indices_2d[i] < len(self.masks_2d)
                    masks_2d.append(self.masks_2d[self.indices_2d[i]] == (i + 1))
                masks_2d = torch.stack(masks_2d, dim=0).float()
        else:
            assert 0 <= s and e <= len(self.masks_2d)
            masks_2d = self.masks_2d[s:e].float()
        inter = F.linear(masks, masks_2d)  # shape: [K, N], do not need *view_mask
        # print(inter.shape, now_area.shape, now_area_2.shape, (now_area_2 - now_area).abs().max())
        area_2d = torch.sum(masks_2d, -1)  # shape: [Nv] 可以预处理, do not need *view_mask
        # match_score = 2. * inter / (area_3d[:, None] + area_2d[None, :]).clamp_min(eps)  # shape: [K, Nv]
        match_score = inter / (area_3d[:, None] + area_2d[None, :] - inter).clamp_min(eps)  # shape: [K, Nv]
        # assert 0 - eps <= match_score.min() and match_score.max() <= 1. + eps, f"{match_score.aminmax()}"
        # print('match_score:', utils.show_shape(match_score), match_score.isnan().any())
        losses['match'] = 1. - (match_score * P).sum(dim=0).mean()
        # losses['match'] = ((1. - match_score) * P).sum(dim=0).mean()
        # shape = (match_score.shape[0], e - s, e - s)
        # diff = F.mse_loss(match_score[:, None, :].expand(shape), A[None, s:e, s:e].expand(shape), reduction='none')
        # diff = diff.sum(dim=-1)  # [K, Nv]
        # losses['match'] = (diff * P).sum(dim=0).mean()
        # matched = diff.argmin(dim=0)
        # losses['match'] = F.cross_entropy(logits[s:e], matched) + \
        #      diff[matched, torch.arange(e - s, device=self.device)].mean()
        if timer is not None:
            timer.log('match score')
        # all 2D masks ~ all 3D masks
        if weights.get('view', 0) > 0:
            # print(match_score.shape, masks.shape, self.M, e - s)
            idx3d, idx2d = linear_sum_assignment((1 - match_score).detach().cpu().numpy())  # 二分匹配
            idx3d, idx2d = utils.tensor_to(idx3d, idx2d, device=match_score.device)
            # idx3d = match_score.argmax(dim=0)
            # idx2d = torch.arange(match_score.shape[1], device=idx3d.device)
            assert idx2d.shape == idx3d.shape
            score_ci = match_score[idx3d, idx2d]

            seen = area_3d / torch.sum(masks, -1).clamp_min(eps)
            no_match = torch.ones(masks.shape[0], device=masks.device, dtype=torch.bool)
            no_match[idx3d] = 0
            losses['view'] = ((seen * scores * no_match).sum() + (1 - score_ci * scores[idx3d]).sum()) / len(masks)
            if timer is not None:
                timer.log('view')

        if weights.get('vm', 0) > 0:
            losses['vm'] = self.loss_view_mask(match_score, scores)
            if timer is not None:
                timer.log('view-masks')
        if weights.get('mv', 0) > 0:
            losses['mv'] = self.loss_mask_view(match_score, scores)
            if timer is not None:
                timer.log('masks-view')
        # Tree loss
        if weights.get('tree', 1) > 0:
            losses['tree'] = self.loss_tree(masks, scores)
            if timer is not None:
                timer.log('tree loss')
        return losses

    def run(
        self,
        epochs=10000,
        K: int = None,
        gnn: pyg_nn.GCN = None,
        A: Tensor = None,
        X: Tensor = None,
        topP=False,
        topP_start=0,
        weights: dict = None,
        print=print,
        use_amp=False,
    ):
        # torch.cuda.empty_cache()
        # torch.set_anomaly_enabled(True)
        # print('[Tree3D] GPU:', utils.get_GPU_memory())
        # if self.view_masks is None or (N_view > 0 and N_view != self.N_view):
        #     self.load_2d_results(self.save_root, N_view)
        assert self.masks_view is not None, f"self.masks_view is {self.masks_view}"
        print('[Tree3D] GPU:', utils.get_GPU_memory())
        K = 2 * self.Lmax if K is None else K
        edge_weight = None
        if gnn is None:
            S = nn.Parameter(torch.randn((self.M, K), device=self.device))
            edges = None
        else:
            assert A is not None and X is not None
            # assert A.shape == (self.M, self.M) and X.shape[0] == self.M
            A_ = A.clone()
            A_[:self.M, :self.M] *= A_[:self.M, :self.M].ge(0.5)
            edges = torch.nonzero(A_).T
            if A.dtype.is_floating_point:
                edge_weight = A_[edges[0], edges[1]]
            S = None
            del A_
        node_score = nn.Parameter(torch.randn((K,), device=self.device))
        print('[Tree3D] GPU:', utils.get_GPU_memory())
        # nn.init.normal_(S)
        if gnn is None:
            opt = torch.optim.Adam([S, node_score], lr=1e-3)
        else:
            opt = torch.optim.Adam(list(gnn.parameters()) + [node_score], lr=1e-3)
        grad_scaler = torch.cuda.amp.GradScaler() if use_amp else None
        # opt = torch.optim.Adam(gnn.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, 1e-6)
        meter = DictMeter()
        timer = utils.TimeWatcher()
        if weights is None:
            weights = {}
        weights = {**dict(match=1, view=1, mv=1, recon=1, t2d=1, tree=0.1, vm=1), **weights}
        print(f'loss weights: {weights}')
        with torch.enable_grad():
            for epoch in range(epochs):
                timer.start()
                view_index = random.randrange(self.V)
                opt.zero_grad()
                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    if gnn is not None:
                        if gnn.supports_edge_weight:
                            S = gnn(X.float(), edges, edge_weight=edge_weight)
                        elif gnn.supports_edge_attr and edge_weight is not None:
                            S = gnn(X.float(), edges, edge_attr=edge_weight[:, None])
                        else:
                            S = gnn(X.float(), edges)
                        if timer is not None:
                            timer.log('gnn')
                loss_dict = self.calc_losses(
                    S[:self.M].float(),
                    node_score,
                    view_index,
                    A,
                    progress=((epoch - topP_start) / (epochs - topP_start)) if topP else 1.,
                    timer=timer,
                    weights=weights,
                )
                total_loss = utils.sum_losses(loss_dict, weights)
                meter.update(loss_dict)
                if use_amp:
                    grad_scaler.scale(total_loss).backward()
                    grad_scaler.step(opt)
                    grad_scaler.update()
                else:
                    total_loss.backward()
                    opt.step()
                lr_scheduler.step()
                if timer is not None:
                    timer.log('update')
                if (epoch + 1) % 100 == 0:
                    print(f"[Tree3D] Epoch {epoch + 1}: loss={total_loss.item():.6f}, {meter.average}")
                    meter.reset()
                    if self.debug:
                        break
                # break
                # return
        if timer is not None:
            print(timer)
        with torch.no_grad():
            scores = node_score.detach().sigmoid()
            self.scores, indices = torch.sort(scores, descending=True)
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                if gnn is not None:
                    gnn.eval()
                    if gnn.supports_edge_weight:
                        S = gnn(X.float(), edges, edge_weight=edge_weight)[:self.M]
                    elif gnn.supports_edge_attr and edge_weight is not None:
                        S = gnn(X.float(), edges, edge_attr=edge_weight[:, None])[:self.M]
                    else:
                        S = gnn(X.float(), edges)[:self.M]
            if topP:
                values, indices = torch.topk(S, k=1, dim=1)
                P = torch.scatter(torch.zeros_like(S), 1, indices, values.softmax(dim=1))
            else:
                P = S.float().softmax(dim=1)[:, indices]  # shape: (N, C)
            # self.scores = self.calc_comm_prob(P).mean(dim=0)
            masks = self._get_masks(P.T)  # [C, nF]
        self.masks = F.pad(masks >= 0.5, (1, 0))
        self.masks_area = torch.sum(self.masks[:, 1:].float(), -1)
        self.scores *= self.masks_area > 0  # remove empty mask
        self.cnt = 0
        self.first[0] = -1
        self.resize(K + 1)
        for i in range(self.masks.shape[0]):
            index = self.node_new()
            if self.scores[i] >= self.score_threshold:
                self.insert(index, 0)

    def get_level(self, aux_data: dict = None, root=0, depth=1, include_faces=False):
        results = self.get_levels(aux_data, root, depth, include_faces)
        return results[depth] if len(results) > depth else torch.tensor([])

    def get_levels(self, aux_data: dict = None, root=0, depth=-1, include_faces=False):
        levels = super().get_levels(root=root, depth=depth)
        # print(f'[Tree3D] levels without auxdata:', levels)
        if aux_data is not None:
            levels = [level.new_tensor([x for x in level if x.item() in aux_data]) for level in levels]
        levels = [level for level in levels if level.numel() > 0]
        return levels

    def save(self, save_path):
        torch.save({name: getattr(self, name, None) for name in self._save_list}, save_path)
        print('[Tree3D] Save results to', save_path)

    def load(self, save_path):
        for name, value in torch.load(save_path, map_location=self.device).items():
            setattr(self, name, value)
        print('[Tree3D] load results from:', save_path)
        # self.print_tree()

    def to(self, device):
        super().to(device)
        self.scores = self.scores.to(device)
        # self.area = self.area.to(device)
        self.masks = self.masks.to(device)
        return self

    def node_rearrange(self, indices=None):
        indices, new_indices = super().node_rearrange(indices)
        N = self.masks.shape[0]
        temp = indices.new_full((N + 1,), N + 10)
        temp[indices] = torch.arange(len(indices), dtype=indices.dtype, device=indices.device)
        masks_order = torch.argsort(temp)[1:] - 1
        assert 0 <= indices.min() and indices.max() <= N
        assert 0 <= masks_order.min() and masks_order.max() < N, f"{masks_order.aminmax()} vs {N}"
        self.masks = self.masks[masks_order]
        self.scores = self.scores[masks_order]
        return indices, new_indices


def load_DNeRF_cameras_and_images(root: Path, camera_file, img_dir):
    with root.joinpath(camera_file).open('r') as f:
        meta = json.load(f)
    cams = []
    paths = []
    times = []
    for i in range(len(meta['frames'])):
        frame = meta['frames'][i]
        cams.append(np.array(frame['transform_matrix'], dtype=np.float32))
        paths.append(root.joinpath(img_dir, frame['file_path'] + '.png'))
        times.append(frame['time'] if 'time' in frame else float(i) / (len(meta['frames']) - 1))
    fovx = float(meta["camera_angle_x"])
    Tv2w = torch.from_numpy(np.stack(cams, axis=0))
    times = torch.tensor(times, dtype=torch.float)
    Tv2w = ops_3d.convert_coord_system(Tv2w, 'opengl', 'colmap', inverse=True)
    images = np.stack([utils.load_image(img_path) for img_path in paths])
    print(images.shape)
    return images, Tv2w, times, fovx


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class Camera_2(nn.Module):
    def __init__(
        self, colmap_id, R, T, FoVx, FoVy,
        # image, gt_alpha_mask,
        image_size,
        image_name, uid,
        trans=np.array([0.0, 0.0, 0.0]), scale=1.0,
        data_device="cpu"
    ):
        super(Camera_2, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.image_width = image_size[0]
        self.image_height = image_size[1]
        # self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        # self.image_width = self.original_image.shape[1]
        # self.image_height = self.original_image.shape[0]
        #
        # if gt_alpha_mask is not None:
        #     self.original_image *= gt_alpha_mask.to(self.data_device)
        # else:
        #     self.original_image *= torch.ones((self.image_height, self.image_width, 1), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image  # .permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1)  # .permute(2, 0, 1)


WARNED = False


def loadCam(downscale, uid, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if downscale in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * downscale)), round(orig_h / (resolution_scale * downscale))
    else:  # should be a type that converts to float
        if downscale == -1:
            max_size = 1024
            if orig_w > max_size:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / max_size
            else:
                global_down = 1
        else:
            global_down = orig_w / downscale

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    # resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    #
    # gt_image = resized_image_rgb[..., :3]
    # loaded_mask = None
    #
    # if resized_image_rgb.shape[1] == 4:
    #     loaded_mask = resized_image_rgb[..., 3:4]
    # print(resolution)
    return Camera_2(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
        FoVx=cam_info.FovX, FoVy=cam_info.FovY,
        # image=gt_image, gt_alpha_mask=loaded_mask,
        image_size=resolution,
        image_name=cam_info.image_name, uid=uid, data_device='cpu')


def cameraList_from_camInfos(cam_infos, resolution_scale, downscale):
    camera_list = []

    for idx, c in enumerate(cam_infos):
        camera_list.append(loadCam(downscale, idx, c, resolution_scale))

    return camera_list


def load_colmap_cameras(root: Path, colmap_dir='sparse/0', images_dir='images'):
    colmap_dir = root.joinpath(colmap_dir)
    images_dir = root.joinpath(images_dir)
    if colmap_dir.joinpath('images.bin').exists():
        cam_extrinsics = colmap_util.read_extrinsics_binary(colmap_dir.joinpath('images.bin'))
    else:
        cam_extrinsics = colmap_util.read_extrinsics_text(colmap_dir.joinpath('images.txt'))

    if colmap_dir.joinpath('cameras.bin').exists():
        cam_intrinsics = colmap_util.read_intrinsics_binary(colmap_dir.joinpath('cameras.bin'))
    else:
        cam_intrinsics = colmap_util.read_intrinsics_binary(colmap_dir.joinpath('cameras.txt'))

    cam_infos_unsorted = colmap_util.readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=images_dir)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    # nerf_normalization = getNerfppNorm(cam_infos)
    cameras = cameraList_from_camInfos(cam_infos, 1.0, -1)
    return cameras
    # fovx = cameras[0].FoVx
    # fovy = cameras[0].FoVy
    # image_size = (cameras[0].image_width, cameras[0].image_height)
    # # point_cloud = scene_info.point_cloud
    # Tw2v = torch.stack([camera.world_view_transform.transpose(-1, -2) for camera in cameras])
    # Tv2c = torch.stack([camera.projection_matrix.transpose(-1, -2) for camera in cameras])
    # return fovx, fovy, Tw2v, Tv2c, image_size


@torch.no_grad()
def get_tri_id(
    model: GaussianSplatting,
    cameras: List[Camera_2],
    topk=10,
    images: Tensor = None,
):
    model.tanfovx = math.tan(0.5 * cameras[0].FoVx)
    model.tanfovy = math.tan(0.5 * cameras[0].FoVy)
    avg_PSNR = 0
    pc_indices, pc_weights = [], []
    weights_max = []
    num_points = model.points.shape[0]
    rendered_images = []
    print(utils.get_GPU_memory())
    for i in tqdm(range(len(cameras)), desc='render images'):
        outputs_i = model.render(info={
            'size': (cameras[i].image_width, cameras[i].image_height),
            'Tw2v': cameras[i].world_view_transform.transpose(-1, -2).cuda(),
            'Tw2c': cameras[i].full_proj_transform.transpose(-1, -2).cuda(),
            'campos': cameras[i].camera_center.cuda(),
        },
            background=torch.ones(3).cuda()
        )
        image_i = outputs_i['images'][0].clamp(0, 1)
        # plt.imshow(utils.as_np_image(image_i))
        # plt.show()
        # exit()
        rendered_images.append(image_i.cpu())
        if images is not None:
            gt_image = torch.from_numpy(images[i]).cuda() / 255.
            avg_PSNR += -10 * math.log10(F.mse_loss(image_i, gt_image[..., :3]).item())

        indices_i, weights_i = topk_weights(topk, outputs_i['buffer'])
        # print(indices_i.shape, weights_i.shape)
        # print(weights_i[:, :, 0][indices_i[:, :, 0] > 0])
        # print(weights_i[:, :, 0][indices_i[:, :, 0] > 0].aminmax())
        # for i in range(weights_i.shape[-1] - 1):
        #     assert torch.all(weights_i[:, :, i] >= weights_i[:, :, i + 1])
        pc_weights.append(weights_i.cpu())
        pc_indices.append(indices_i.cpu())
        temp = torch.zeros(num_points, device=torch.device('cuda'))
        temp.scatter_reduce_(0, indices_i.view(-1).long().clamp(0), weights_i.view(-1), 'sum')
        weights_max.append(temp.cpu())
        if i == 0:
            print(utils.show_shape(outputs_i, indices_i, weights_i))
        del outputs_i, image_i
    weights_max = torch.amax(torch.stack(weights_max), dim=0)
    if images is not None:
        print('PSNR:', avg_PSNR / len(images))
    pc_weights, pc_indices = torch.stack(pc_weights), torch.stack(pc_indices)
    pc_weights = pc_weights.permute(0, 3, 1, 2)
    pc_indices = pc_indices.permute(0, 3, 1, 2)
    print(f'pixel corresponding points: indices {pc_indices.shape}, weights: {pc_weights.shape}')
    rendered_images = torch.stack(rendered_images)
    return pc_weights.cpu(), pc_indices.cpu(), weights_max.cpu(), rendered_images


def run_2d_tree_segmentation(args, images: Tensor, save_dir: Path):
    # features = []
    results = []
    images = (images[:, :, :, :3] * 255).to(torch.uint8).cpu().numpy()
    print(images.shape)
    N, H, W, _ = images.shape
    # first stage
    for i in tqdm(range(N), desc='2d tree seg'):
        if save_dir.joinpath(f'img_{i:03d}.jpg').exists():
            continue
        predictor = get_predictor(args)
        predictor.set_image(images[i])
        results = predictor.tree_generate(
            image=images[i],
            points_per_side=args.points_per_side,
            points_per_update=args.points_per_update,
            min_mask_region_area=args.min_area,
            max_steps=args.max_steps,
            in_threshold=args.in_threshold,
            in_thre_area=args.in_area_threshold,
            union_threshold=args.union_threshold,
            device=torch.device('cuda'))
        results.post_process()

        masks = results.masks.cpu().numpy()
        levels = results.get_levels()
        for k, level_k in enumerate(levels):
            if k == 0:
                continue
            mask = masks[level_k.cpu().numpy() - 1].astype(np.uint8)
            mask = np.max(mask * np.arange(1, mask.shape[0] + 1)[:, None, None], axis=0).astype(np.uint8)
            # plt.imshow(mask)
            # plt.show()
            colors = np.array([[1, 1, 1]] + sns.color_palette(n_colors=len(level_k)))
            mask_i = Image.fromarray(mask, mode='P')
            mask_i.putpalette((colors * 255).astype(np.uint8))
            mask_i.save(save_dir.joinpath(f'img_{i:03d}_level_{k:02d}.png'))
        utils.save_image(save_dir / f'img_{i:03d}.jpg', images[i])
    return


def options():
    parser = argparse.ArgumentParser()
    predictor_options(parser)
    parser.add_argument('-o', '--output', default='./results/3D_GS/bicycle')
    return parser.parse_args()


def main():
    args = options()
    model = load_model()
    # vis_model(model)
    # return
    save_dir = Path(args.output).expanduser()
    save_dir.mkdir(exist_ok=True, parents=True)
    tree_seg_pc = GSTreeSegmentation(model.points.shape[0], device=torch.device('cuda'), debug=False)
    save_path = save_dir.joinpath('bicycle.tree3dv2')
    if 1 and save_path.exists():
        tree_seg_pc.load(save_path)
        print(f'load tree_pc from {save_path}')
        tree_seg_pc.print_tree()
        points = model.points
        for level, nodes in enumerate(tree_seg_pc.get_levels()):
            if level == 0:
                continue
            print(nodes)
            colors = np.concatenate([np.array([[0.1, 0.1, 0.1]]), sns.color_palette(n_colors=len(nodes))], 0)
            print(utils.show_shape(tree_seg_pc.masks, colors))
            masks = tree_seg_pc.masks[nodes - 1, 1:]
            print(utils.show_shape(masks))
            masks = (masks * torch.arange(len(nodes), device=masks.device)[:, None])
            print(utils.show_shape(masks))
            masks = masks.amax(dim=0)
            print(utils.show_shape(masks))
            colors = colors[masks.cpu().numpy()]
            print(utils.show_shape(colors, points))
            utils.save_point_clouds(save_dir.joinpath(f"./level_{level}.ply"), points, colors)
    else:
        # images, Tv2w, times, fovx = load_DNeRF_cameras_and_images(
        #     Path('~/data/NeRF/D_NeRF/lego').expanduser(), 'transforms_train.json', '')
        # fovx, fovy, Tw2v, Tv2c, image_size = load_colmap_cameras(Path('~/data/NeRF/Mip360/bicycle').expanduser())
        cameras = load_colmap_cameras(Path('~/data/NeRF/Mip360/bicycle').expanduser())
        # pc_weights, pc_indices, w_max = get_tri_id(model, images, Tv2w, fovx, topk=10)
        pc_weights, pc_indices, w_max, images = get_tri_id(model, cameras, topk=10)
        print('get_tri_id, GPU: {:.3f}/{:.3f}'.format(*utils.get_GPU_memory()))
        # if len(list(save_dir.glob('*.png'))) == 0:
        run_2d_tree_segmentation(args, images, save_dir)
        tree_2d_results = tree_seg_pc.load_2d_results(save_dir)
        del model
        torch.cuda.empty_cache()
        # exit()
        print('load_2d_results, GPU: {:.3f}/{:.3f}'.format(*utils.get_GPU_memory()))
        tree_seg_pc.init_from_2D_reults(tree_2d_results[::2], pc_indices[::2], pc_weights[::2], w_max[::2], pack=True)
        print('init_from_2D_reults, GPU: {:.3f}/{:.3f}'.format(*utils.get_GPU_memory()))
        if save_dir.joinpath('A.pth').exists():
            A = torch.load(save_path.joinpath('A.pth'), map_location='cpu').cuda()
        else:
            A = tree_seg_pc.build_all_graph()
            torch.save(A, save_dir.joinpath('A'))
        print('build_all_graph, GPU: {:.3f}/{:.3f}'.format(*utils.get_GPU_memory()))
        if save_dir.joinpath('X.pth').exists():
            X = torch.load(save_path.joinpath('X.pth'), map_location='cpu').cuda()
        else:
            X, _ = tree_seg_pc.compress_masks(hidden_dims=(32, 128, 256))
            torch.save(X, save_dir.joinpath('X'))
        print('compress_masks, GPU: {:.3f}/{:.3f}'.format(*utils.get_GPU_memory()))
        K = 2 * tree_seg_pc.Lmax
        gnn = pyg_nn.GCN(
            in_channels=X.shape[1],
            hidden_channels=128,
            num_layers=2,
            out_channels=K,
            norm='BatchNorm',
        ).cuda()
        tree_seg_pc.run(A=A, X=X, K=K, gnn=gnn)
        tree_seg_pc.save(save_path)
        tree_seg_pc.print_tree()


if __name__ == '__main__':
    main()
