import math
import os
import random
from pathlib import Path
from typing import List, Tuple, Optional, Sequence, Dict

import torch
import torch.amp
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore
from torch_scatter import scatter
import torch_geometric.nn as pyg_nn

from scipy.optimize import linear_sum_assignment
from tree_segmentation.extension import Mesh, utils
from tree_segmentation.extension import ops_3d
from tree_segmentation import TreeStructure, MaskData, Tree2D, extension as ext


class Tree3D(TreeStructure):

    def __init__(
        self,
        mesh: Mesh,
        device=None,
        in_threshold=0.9,
        in_thre_area=10,
        union_threshold=0.1,
        ignore_area=50,
        verbose=1,
        momentum=0.9,
    ):
        self.device = device
        # mesh
        self.mesh = mesh
        v3 = mesh.v_pos[mesh.f_pos]  # shape: (F, 3, 3)
        self.area = torch.cross(v3[:, 0] - v3[:, 1], v3[:, 0] - v3[:, 2], dim=-1).norm(dim=-1) * 0.5
        assert torch.all(self.area.ge(0))
        self.num_faces = mesh.f_pos.shape[0]
        # tree structure
        self.face_parent = torch.empty(self.num_faces + 1, device=device, dtype=torch.int)
        self.score = torch.empty((1, self.num_faces + 1), device=device, dtype=torch.float)
        self.score_node = torch.empty((1, 2), device=device, dtype=torch.int)
        super().__init__(1, device=device, verbose=verbose)
        # merge parameters
        self.threshold_in = in_threshold
        self.threshold_in_area = in_thre_area
        self.threshold_union = union_threshold
        self.threshold_score = 0.5
        self.threshold_score_node = 0.5
        self.momentum = momentum
        self.ignore_area = ignore_area
        self.pad_length = 10  # add 10 empty nodes when enlarge nodes
        self.conflict_nodes = [[]]

    def reset(self):
        self.face_parent.fill_(-1)
        self.score.fill_(-1)
        self.score_node.fill_(5.)
        self.conflict_nodes = [[]]
        super().reset()

    def resize(self, N: int):
        N, M = super().resize(N)
        if M > N:
            self.score = self.score[:N]
            self.score_node = self.score_node[:N]
        elif M < N:
            self.score = torch.constant_pad_nd(self.score, (0, 0, 0, N - M), -1)
            self.score_node = torch.constant_pad_nd(self.score_node, (0, 0, 0, N - M), 5.)
            # print(utils.show_shape(self.score, self.node_score))

    def node_new(self):
        if self.cnt + 1 == len(self.parent):
            self.resize(self.cnt + 1 + self.pad_length)
        index = super().node_new()
        self.score[index].fill_(-1)
        self.score_node[index].fill_(5)
        return index

    # def node_delete(self, idx: int, move_children=False):
    #     # TODO: move face_parent
    #     return super().node_delete(idx, move_children)

    def node_rearrange(self, indices=None):
        if indices is None:
            conflict_nodes = [torch.tensor(x, dtype=torch.int, device=self.device) for x in self.conflict_nodes]
            indices = torch.cat(self.get_levels() + conflict_nodes, dim=0)
        indices, new_indices = super().node_rearrange(indices)
        num = len(indices)
        self.face_parent = new_indices[self.face_parent + 1]
        self.score[:num] = self.score[indices]
        self.score_node[:num] = self.score_node[indices]
        self.conflict_nodes = [[new_indices[i + 1].item() for i in L] for L in self.conflict_nodes]
        return indices, new_indices

    def proposal_camera_pose(self, radius_range=(2.5, 3.), elev_range=(0, 180), azim_range=(0, 360), device=None):
        """根据现状提议相机位姿"""
        if device is None:
            device = self.device
        radius = torch.rand(1, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]
        thetas = torch.rand(1, device=device) * (elev_range[1] - elev_range[0]) + elev_range[0]
        phis = torch.rand(1, device=device) * (azim_range[1] - azim_range[0]) + azim_range[0]

        eye = ops_3d.coord_spherical_to(radius, thetas.deg2rad(), phis.deg2rad()).to(device)
        return ops_3d.look_at(eye, torch.zeros_like(eye))

    def proposal_camera_pose_uniform(self, num=1, radius_range=(2.5, 3.), elev_range=(0, 180), azim_range=(0, 360)):
        cos_theta_range = (math.cos(math.radians(elev_range[0])), math.cos(math.radians(elev_range[1])))
        phi_range = (math.radians(azim_range[0]), math.radians(azim_range[1]))
        phis = torch.rand((num,)) * (phi_range[1] - phi_range[0]) + phi_range[0]
        thetas = torch.arccos(torch.rand((num,)) * (cos_theta_range[1] - cos_theta_range[0]) + cos_theta_range[0])
        radius = torch.rand((num,)) * (radius_range[1] - radius_range[0]) + radius_range[0]

        eye = ops_3d.coord_spherical_to(radius, thetas, phis).to(self.device)
        return ops_3d.look_at(eye, torch.zeros_like(eye))

    def proposal_camera_pose_spherical_grid(self, num=1, radius_range=(2.5, 2.5), device=None):
        """球面上的均匀格点"""
        seq = torch.arange(num, device=device)
        c = (math.sqrt(5) - 1) * math.pi
        z = ((seq * 2 - 1) / num - 1).clamp(-1, 1)
        x = torch.sqrt(1 - z * z) * (seq * c).cos()
        y = torch.sqrt(1 - z * z) * (seq * c).sin()
        eye = torch.stack([x, y, z], dim=-1)
        eye = eye * (torch.rand((num, 1), device=device) * (radius_range[1] - radius_range[0]) + radius_range[0])
        return ops_3d.look_at(eye, torch.zeros_like(eye))

    def proposal_camera_pose_cycle(self,
                                   num=1,
                                   radius_range=(2.5, 3.),
                                   elev_range=(0, 180),
                                   azim_range=(0, 360),
                                   device=None):
        theta_range = (math.radians(elev_range[0]), math.radians(elev_range[1]))
        phi_range = (math.radians(azim_range[0]), math.radians(azim_range[1]))
        # phase 1
        num1 = int(0.5 * num)
        thetas_1 = (torch.arange(num1) + 0.5) / num1 * (theta_range[1] - theta_range[0]) + theta_range[0]
        phis_1 = torch.ones((num1,)) * (phi_range[1] + phi_range[0]) * 0.5
        # phase 2
        num2 = num - num1
        thetas_2 = torch.ones((num2,)) * (theta_range[1] + theta_range[0]) * 0.5
        phis_2 = (torch.arange(num2) + 0.5) / num2 * (phi_range[1] - phi_range[0]) + phi_range[0]

        thetas = torch.cat([thetas_1, thetas_2])
        phis = torch.cat([phis_1, phis_2])
        radius = torch.rand((num,)) * (radius_range[1] - radius_range[0]) + radius_range[0]
        eye = ops_3d.coord_spherical_to(radius, thetas, phis).to(device)
        return ops_3d.look_at(eye, torch.zeros_like(eye))

    def proposal_points(self, tri_id: Tensor, num_points=256):
        """在当前相机位姿下提议SAM的promt points
        Args:
            tri_id: shape: [H, W], offset by one, 0 --> none mesh
            num_points:
        """
        # find un-segment triangles-
        # parent_faces = self.parent[tri_id]
        # print(parent_faces.shape)
        # undeal_index = torch.nonzero(parent_faces == 0 & tri_id > 0)
        unique_tri_ids = torch.unique(tri_id, sorted=True)
        if unique_tri_ids[0] == 0:
            unique_tri_ids = unique_tri_ids[1:]
        parent_faces = self.parent[unique_tri_ids]
        num_unsegment = parent_faces.shape[0] - (parent_faces >= 0).sum().item()
        if num_unsegment > 0:
            sample_face_indices = unique_tri_ids[parent_faces < 0]
            return self.sample_points_by_area(sample_face_indices, num_points)
        # next level sample
        raise NotImplementedError()

    def sample_points_by_area(self, sample_face_indices: Tensor, num_points: int):
        # print('sample_faces', sample_face_indices.shape)
        assert 1 <= sample_face_indices.min() and sample_face_indices.max() <= self.num_faces
        # print('sample_weighs:', *sample_face_indices.aminmax(), self.area.shape)
        sample_weights = self.area[sample_face_indices - 1]
        sample_weights = sample_weights / sample_weights.sum()
        # print('sample_weights', sample_weights.shape, sample_weights.sum())
        sample_rand = torch.rand(num_points, device=sample_face_indices.device)
        sample_idx = torch.searchsorted(sample_weights.cumsum(dim=0), sample_rand)

        bc = torch.rand((num_points, 3), device=sample_face_indices.device)  # 重心坐标
        bc = bc / bc.sum(dim=-1, keepdim=True)
        verts = self.mesh.v_pos[self.mesh.f_pos[sample_face_indices[sample_idx]]]
        sample_points = torch.sum(verts * bc[:, :, None], dim=1)
        return sample_points

    def get_aux_data(self, tri_id: Tensor):
        indices = self.face_parent[tri_id]
        aux_data = {}
        for nodes in reversed(self.get_levels()):
            for x in nodes:
                mask = indices.eq(x)
                area = mask.sum().item()
                if area >= self.ignore_area:
                    aux_data[x.item()] = (mask, area)
                indices[mask] = self.parent[x].item()
        mask = tri_id > 0
        aux_data[0] = (mask, mask.sum().item())
        aux_data['tri_id'] = tri_id
        # print('[Tree3D] get_aux_data:', utils.show_shape(aux_data))
        # print(f'There are {len(aux_data) - 1} segmented masks')
        return aux_data

    def get_level(self, aux_data: dict = None, root=0, depth=1, include_faces=False):
        results = self.get_levels(aux_data, root, depth, include_faces)
        return results[depth] if len(results) > depth else torch.tensor([])

    def get_levels(self, aux_data: dict = None, root=0, depth=-1, include_faces=False):
        levels = super().get_levels(root=root, depth=depth)
        # print(f'[Tree3D] levels without auxdata:', levels)
        if aux_data is not None:
            levels = [level.new_tensor([x for x in level if x.item() in aux_data]) for level in levels]
        if include_faces:
            faces = torch.unique(aux_data['tri_id'])
            parents = self.face_parent[faces]
            levels.append(torch.tensor([], dtype=torch.int, device=self.device))
            for i, level in reversed(list(enumerate(levels))):
                levels[i + 1] = torch.cat([levels[i + 1]] + [faces[parents == x] for x in level])
        levels = [level for level in levels if level.numel() > 0]
        # if self.verbose > 1 or (self.verbose == 1 and depth < 0):
        #     print(f'[Tree3D] get {len(levels)} levels')
        return levels

    def update(self, mask_data: MaskData, aux_data: dict, one_batch=True):
        for i in range(len(mask_data['masks'])):
            if self.verbose > 0:
                print('[Tree3D]', '=' * 10, f'mask [{i + 1}]', '=' * 10)
            self.update_one_mask(aux_data, mask_data['masks'][i].to(self.device))
            # print('[Tree3D] after update:', utils.show_shape(aux_data))
            # print('[Tree3D] levels:', self.get_levels(aux_data))
        # self.print_tree()
        if one_batch:
            for nodes in self.get_levels(aux_data):
                for now in nodes:
                    self.score_node[now, 0] += 1
            self.update_tree()
        print('num_nodes:', sum(len(x) for x in self.get_levels()), 'num conflict:',
              sum(len(x) for x in self.conflict_nodes), self.cnt)
        # self.node_rearrange()
        # self.print_tree()

    def update_one_mask(
        self,
        aux_data: dict,
        mask: Tensor,
        area=None,
        root=0,
        mask_new: Tensor = None,
        area_new=0,
        level=0,
    ):
        if root == 0:
            area = mask.sum().item()
            # mask = self.preprocess_mask(mask, aux_data)
            # mask_area = mask.sum().item()
            if area <= self.ignore_area:
                if self.verbose > 0:
                    print(f"[Tree3D] mask area is too small: {area} < {self.ignore_area}, skip")
                return False
            mask_j, area_j = aux_data[root]
            mask = torch.logical_and(mask, mask_j)
            inter = mask.sum().item()
            # union = area_j + mask_area - inter
            if inter / area_j >= self.threshold_in or area_j - inter <= self.threshold_in_area:  # foreground
                if self.verbose > 0:
                    print(f'[Tree3D] whole scene, skip')
                return False
            if 1 - inter / area >= self.threshold_in or inter <= self.threshold_in_area:  # background
                if self.verbose > 0:
                    print(f"[Tree3D] background, skip")
                return False
            area = inter
            mask_new = (self.face_parent[aux_data['tri_id']] == -1) & mask
            area_new = mask_new.sum().item()
            if area_new / area >= self.threshold_in:
                now = self.node_new()
                if self.verbose > 0:
                    print(f'[Tree3D] new area: {now}')
                self.node_insert(now, root)
                aux_data[now] = (mask, area)
                self._update_score(aux_data, mask, now)
                self._update_face_parent(aux_data, mask, now, mask_new, area_new)
                return True

        nodes_in_i = []
        nodes_union = []
        for j in self.get_children(root):
            if j not in aux_data:
                continue
            mask_j, area_j = aux_data[j]
            inter = torch.logical_and(mask, mask_j).sum().item() + area_new  # consider new mask as a part of mask_j
            area_j += area_new
            union = area_j + area - inter
            if inter / area >= self.threshold_in or area - inter <= self.threshold_in_area:  # i in j
                if inter / area_j >= self.threshold_in or area_j - inter <= self.threshold_in_area:  # j == i
                    if self.verbose > 0:
                        print(f'[Tree3D] mask equal to {j}')
                    self._update_score(aux_data, mask, j)
                    self._update_face_parent(aux_data, mask, j, mask_new, area_new)
                    return True
                else:
                    if self.verbose > 0:
                        print(f"[Tree3D] mask belong to {j}")
                    return self.update_one_mask(aux_data, mask, area, j, mask_new, area_new, level=level + 1)
            elif inter / area_j >= self.threshold_in:  # j in i
                nodes_in_i.append(j)
            elif inter / union >= self.threshold_union:
                nodes_union.append((j, f"{inter / union:.2%}"))
            else:  # no intersect
                pass
        if len(nodes_union) > 0:
            if self.verbose > 0:
                print(f"[Tree3D] union with {nodes_union}")
            is_same = False
            while level >= len(self.conflict_nodes):
                self.conflict_nodes.append([])
            for j in self.conflict_nodes[level]:
                mask_j = (self.score[j] > self.threshold_score)[aux_data['tri_id']]
                area_j = mask_j.sum().item() + area_new
                inter = torch.logical_and(mask, mask_j).sum().item() + area_new
                if inter / area >= self.threshold_in or area - inter <= self.threshold_in_area:
                    if inter / area_j >= self.threshold_in or area_j - inter <= self.threshold_in_area:
                        is_same = True
                        self._update_score(aux_data, mask, j)
                        break
            if not is_same:
                now = self.node_new()
                self._update_score(aux_data, mask, now)
                self.conflict_nodes[level].append(now)
            return False
        now = self.node_new()
        self.node_insert(now, root)
        self._update_score(aux_data, mask, now)
        self._update_face_parent(aux_data, mask, now, mask_new, area_new)
        aux_data[now] = (mask, area)
        if self.verbose > 0:
            print(f"[Tree3D] insert mask {now} as child of {root}")
        for j in nodes_in_i:
            self.node_move(j, now)
            if self.verbose > 0 and (self.verbose > 1 or j > self.num_faces):
                print(f'[Tree3D] move {j} as child of mask {now}')
        return True

    def _update_score(self, aux_data, mask: Tensor, index: int):
        if 'tri_unique' not in aux_data:
            aux_data['tri_unique'] = aux_data['tri_id'].unique(return_counts=True)
        tf, tc = aux_data['tri_unique']
        faces, counts = aux_data['tri_id'][mask].unique(return_counts=True)
        tmp = torch.zeros_like(self.score[index])
        tmp[faces] = counts.float()
        tmp = tmp[tf] / tc.float()
        now = self.score[index, tf]
        self.score[index][tf] = torch.where(now < 0, tmp, torch.lerp(tmp, now, self.momentum))
        self.score[index, 0] = 0
        self.score_node[index, 1] += 1
        return

    def _update_face_parent(self, aux_data, mask: Tensor, index: int, new_mask: Tensor, new_area, threshold=0.5):
        parent = self.parent[index].item()
        faces = torch.unique(aux_data['tri_id'][mask])
        if faces[0] == 0:
            faces = faces[1:]
        face_parent = self.face_parent[faces]
        need_update = (face_parent == parent) | (face_parent == -1)
        if need_update.numel() > 0:
            update_face = faces[need_update]
            self.face_parent[update_face] = index
        if new_area > 0:
            while parent != 0:
                root_mask = aux_data[parent][0] | new_mask
                aux_data[parent] = (root_mask, root_mask.sum().item())
                parent = self.parent[parent].item()

    def update_tree(self, threshold_node_delete=0.1):
        self.node_rearrange()
        node_score = self.score_node[:1 + self.cnt, 1] / self.score_node[:self.cnt + 1, 0]
        node_score[0] = 1e9

        self.parent.fill_(-1)
        self.first.fill_(-1)
        self.last.fill_(-1)
        self.next.fill_(-1)
        self.face_parent.fill_(-1)
        self.conflict_nodes = []
        aux_data = {}

        for i in torch.argsort(node_score, descending=True):
            i = i.item()
            if i == 0:
                continue
            if node_score[i] < threshold_node_delete:
                if self.verbose > 0:
                    print(f'[Tree3D] Delete node {i}, node score {node_score[i]} < threshold {threshold_node_delete}')
                continue
            mask = self.score[i] >= self.threshold_score
            if not torch.any(mask):
                print(f'[Tree3D] deldete node {i}, empty faces')
                continue
            area = (mask[1:] * self.area).sum().item()
            self._update_one_instance(i, mask, area, aux_data)
        self.node_rearrange()

    def _update_one_instance(self, now: int, mask: Tensor, area: float, aux_data: dict, root=0, level=0):
        nodes_in = []
        nodes_union = []
        for j in self.get_children(root):
            mask_j, area_j = aux_data[j]
            inter = ((mask & mask_j)[1:] * self.area).sum().item()
            if inter / area >= self.threshold_in:
                if inter / area_j >= self.threshold_in:
                    self.score_node[j, 0] += self.score_node[now, 0]
                    self.score[j] = 0.5 * (self.score[j] + self.score[now])
                    mask = self.score[j] >= self.threshold_score
                    area = (mask[1:] * self.area).sum().item()
                    aux_data[j] = (mask, area)
                    print(f'[Tree3D] same instance {now}, {j}')
                else:
                    self._update_one_instance(now, mask, area, aux_data, j, level + 1)
                return
            elif inter / area_j >= self.threshold_in:
                nodes_in.append(j)
            elif inter / (area + area_j - inter) >= self.threshold_union:
                nodes_union.append(j)
        if len(nodes_union) > 0:
            while len(self.conflict_nodes) <= level:
                self.conflict_nodes.append([])
            has_same = False
            for j in self.conflict_nodes[level]:
                mask_j, area_j = aux_data[j]
                inter = ((mask & mask_j)[1:] * self.area).abs().sum().item()
                if inter / area >= self.threshold_in and inter / area_j >= self.threshold_in:
                    self.score_node[j, 0] += self.score_node[now, 0]
                    self.score[j] = 0.5 * (self.score[j] + self.score[now])
                    mask = self.score[j] >= self.threshold_score
                    area = (mask[1:] * self.area).sum().item()
                    aux_data[j] = (mask, area)
                    print(f'[Tree3D] same confict instance {now} {j}')
                    has_same = True
                    break
            if not has_same:
                self.conflict_nodes[level].append(now)
                aux_data[now] = (mask, area)
            return
        self.node_insert(now, root)
        aux_data[now] = (mask, area)
        self.face_parent[mask & ((self.face_parent < 0) | (self.face_parent == root))] = now
        for j in nodes_in:
            self.node_move(j, now)

    def preprocess_mask(self, mask: Tensor, aux_data: dict, threshold=0.5):
        """预处理mask: 选择占比大于threshold的所有面片"""
        if 'num_faces' not in aux_data:
            faces, counts = torch.unique(aux_data['tri_id'], return_counts=True)
            aux_data['num_faces'] = {k.item(): v.item() for k, v in zip(faces, counts)}
        faces_on_mask = aux_data['tri_id'][mask]
        num_faces = aux_data['num_faces']
        faces, counts = torch.unique(faces_on_mask, return_counts=True)
        mask_ = torch.zeros_like(mask)
        for fi, area in zip(faces, counts):
            fi, area = fi.item(), area.item()
            if fi == 0:
                continue
            area_all = num_faces[fi]
            if area / area_all >= threshold:
                mask_ |= aux_data['tri_id'] == fi  # extend mask to all pixels of triangle fi
            else:
                pass
                # mask = mask & (aux_data['tri_id'] != fi)
                # print(f'face {fi.item()} is filtered, {area/area_all:.2%}')
        return mask_

    def verify_masks(self, mask_data: MaskData, aux_data: dict):
        for i in range(len(mask_data['masks'])):
            mask = mask_data['masks']
            self.verify_one_mask(mask, aux_data, 0)

    def verify_one_mask(self, mask: Tensor, aux_data: dict, root=0):
        pass

    def compress(self):
        if self.verbose > 0:
            print('[Tree3D] compress')
        levels = self.get_levels()
        face_parent = self.face_parent.clone()
        for level, nodes in reversed(list(enumerate(levels))):
            # print(level, nodes)
            if level == 0 or level == 1:
                continue
            for i in nodes:
                faces_i = face_parent == i
                pa = self.parent[i]
                face_parent[faces_i] = pa
                if self.last[i] < 0 and self.next[i] < 0:  # only one child
                    faces_p = face_parent == pa
                    inter = self.area[(faces_i & faces_p)[1:]].sum().item()
                    union = self.area[(faces_i | faces_p)[1:]].sum().item()
                    if inter / union >= self.threshold_in:
                        self.node_delete(i, move_children=True)
                        if self.verbose > 0:
                            print(f'[Tree3D]: compress {i} and {pa}: {inter / union:.2%}')

        self.node_rearrange()

    def save(self, filename):
        torch.save(
            {
                'parent': self.parent,
                'first': self.first,
                'next': self.next,
                'last': self.last,
                'cnt': self.cnt,
                'face_parent': self.face_parent,
            }, filename)
        if self.verbose > 0:
            print(f"[Tree3D]save now Tree3D to {filename}")

    def load(self, filename):
        if not os.path.exists(filename):
            if self.verbose > 0:
                print(f'[Tree3D] No such file: {filename} to load Tree3D')
            return
        data = torch.load(filename, map_location=self.device)
        self.parent = data['parent']
        self.first = data['first']
        self.next = data['next']
        self.last = data['last']
        self.cnt = data['cnt']
        self.face_parent = data['face_parent']
        if self.verbose > 0:
            print('[Tree3D] load now Tree3D from:', filename)

    def to(self, device):
        super().to(device)
        self.face_parent = self.face_parent.to(device)
        self.area = self.area.to(device)
        return self


# noinspection PyAttributeOutsideInit
class Tree3Dv2(TreeStructure):
    _save_list = ['masks', 'scores', 'parent', 'last', 'next', 'first', 'cnt', 'face_mask']

    def __init__(
        self,
        mesh: Mesh,
        device=None,
        in_threshold=0.8,
        union_threshold=0.1,
        #  min_area=100,
        # in_thres_area=10,
        verbose=0,
    ):
        self.device = device
        # mesh
        self.mesh = mesh
        v3 = mesh.v_pos[mesh.f_pos]  # shape: (F, 3, 3)
        self.area = torch.cross(v3[:, 0] - v3[:, 1], v3[:, 0] - v3[:, 2], dim=-1).norm(dim=-1) * 0.5
        assert torch.all(self.area.ge(0))
        self.num_faces = mesh.f_pos.shape[0]
        # tree results
        self.masks: Tensor = None
        self.masks_area: Tensor = None
        self.scores: Tensor = None
        self.face_mask = None  # mark unseen faces
        super().__init__(1, device=device, verbose=verbose)
        self.ignore_area = 10
        self.score_threshold = 0.5
        self.in_threshold = in_threshold
        # self.in_thres_area = in_thres_area
        self.union_threshold = union_threshold
        # self.min_area = min_area

    def reset(self):
        super().reset()
        ## data
        self.tree2ds = []  # type: List[TreeStructure]
        self.view_infos = []  # type: List[Tuple[Tensor, Tensor]]
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
        # self.X = None  # type: Optional[Tensor]
        # self.A = None  # type: Optional[Tensor]
        # self.view_G = None  # type: Optional[Tensor]
        self.M = 0  # the total number of masks for all views
        self.V = 0  #  the total number of views
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
            inter = torch.sum(mask_i * mask_j * self.area)
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
        if self.masks_area is None:
            self.masks_area = torch.mv(self.masks[:, 1:], self.area)
        while self.cnt < len(self.masks):
            self.insert(self.node_new())
        # self.print_tree()

    @staticmethod
    def get_face_list(tri_id: Tensor, mask: Tensor = None):
        faces, cnts = torch.unique(tri_id if mask is None else (mask * tri_id), return_counts=True)
        if faces[0] == 0:
            faces, cnts = faces[1:], cnts[1:]
        faces = faces - 1
        return faces, cnts

    @torch.no_grad()
    def load_2d_results(
        self,
        save_root: Path = None,
        gt: 'Tree3Dv2' = None,
        tri_ids: Tensor = None,
        N_view=-1,
        background_threshold=0.5,
        pack=False,
    ):
        torch.cuda.empty_cache()
        assert (save_root is None) ^ (gt is None), f"Only can use one of save_root or gt"
        print('[Tree3D] GPU:', utils.get_GPU_memory())
        if save_root is not None:
            seg_2d_files = sorted(list(save_root.glob('*.data')))  # [:50]
            print(f'[Tree3D] There are {len(seg_2d_files)} data')
            if N_view > 0:
                seg_2d_files = seg_2d_files[:N_view]
            data = [torch.load(filename, map_location='cpu') for filename in seg_2d_files]  # type: List[dict]
            print('[Tree3D]', utils.show_shape(data[0]))
        else:
            assert tri_ids is not None
            data = tri_ids
            if N_view > 0:
                data = data[:N_view]

        print(f"[Tree3D] Load {len(data)} views")
        print('[Tree3D] GPU:', utils.get_GPU_memory())

        self._masks_2d_packed = pack
        masks_2d = []
        indices_2d = []
        self.range_2d = []
        indices_view = []
        self.range_view = []
        self.range_level = []
        self.view_infos = []
        self.tree2ds = []
        self.masks_view = torch.zeros((len(data), self.num_faces), dtype=torch.bool, device=self.device)
        self.V = 0
        self.M = 0
        self.Lmax = 0
        temp = torch.zeros(self.num_faces, dtype=torch.float, device=self.device)
        mask_2d = torch.zeros(self.num_faces, dtype=torch.int, device=self.device)
        for vid in range(len(data)):
            if gt is None:
                tri_id = data[vid]['tri_id'].to(self.device)

                tree2d = Tree2D(device=self.device)
                tree2d.load(None, **data[vid]['tree_data'])
                tree2d.remove_background(tri_id.eq(0), background_threshold)
                # tree2d.compress()
            else:
                tri_id = data[vid].to(self.device)
                tree2d = gt.get_2d_tree(tri_id)
            v_faces, v_cnts = self.get_face_list(tri_id)

            masks = tree2d.masks
            # print(f'view: {vid}', utils.get_GPU_memory(), tree2d.cnt, utils.show_shape(masks), tree2d.is_compressed)
            assert masks.ndim == 3 and masks.dtype == torch.bool
            num_masks_start = self.M
            num_levels = 0
            for level, nodes in enumerate(tree2d.get_levels()):
                if level == 0:  # skip root
                    continue
                mask_2d.zero_()
                for i in nodes:
                    faces, cnts = self.get_face_list(tri_id, masks[i - 1])
                    # print(utils.show_shape(faces, cnts, torch.unique(tri_id[masks[i - 1]])))
                    assert 0 <= faces.min() and faces.max() < len(temp), f"view: {vid}, {faces.max()} vs {len(temp)}"
                    temp.zero_()
                    temp[faces.long()] = cnts.float()
                    temp[v_faces] /= v_cnts
                    self.M += 1
                    if pack:
                        mask_2d[temp >= 0.5] = self.M  # FIXME: may area = 0
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
            # t = TreeStructure(tree2d.cnt + 1, device=self.device)
            # t.cnt = tree2d.cnt
            # t.parent = tree2d.parent.clone()
            # t.first = tree2d.first.clone()
            # t.next = tree2d.next.clone()
            # t.last = tree2d.last.clone()
            self.tree2ds.append(self._get_node_relationship(tree2d))
            self.view_infos.append((v_faces, v_cnts))
            self.masks_view[self.V, v_faces] = 1  # cnts.float()
            self.range_view.append((num_masks_start, self.M))
            if pack:
                self.range_level.append((len(masks_2d) - num_levels, len(masks_2d)))
            # indices_view.extend([self.V] * num_masks_v)
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
        self.indices_view = torch.tensor(indices_view, dtype=torch.int32, device=self.device)
        self.masks_view = self.masks_view[:self.V]
        # print(self.range_view, utils.show_shape(self.face_masks))
        if pack:
            indices = torch.nonzero(self.masks_2d)
            self._masks_2d_sp = torch.sparse.FloatTensor(
                torch.stack([self.masks_2d[indices[:, 0], indices[:, 1]] - 1, indices[:, 1]]),
                torch.ones(indices.shape[0], device=indices.device),
                [self.M, self.num_faces],
            )
        else:
            self._masks_2d_sp = None
        self.face_mask = F.pad(self.masks_view.any(0), (1, 0))

        print(f'[Tree3D] view_masks, view_infos[0]: {utils.show_shape(self.masks_view, self.view_infos[0])}')
        print(f"[Tree3D] loaded {self.V} views, {self.M} masks, max_num: {self.Lmax}")
        # print(utils.show_shape(self.face_masks.tensors, self.face_masks.index))
        ## remove unseen faces # TODO: remove unseen faces to reduce compuation
        # seen_faces = self.masks_view.any(dim=0)
        # print(seen_faces.shape)
        # self.area
        print('[Tree3D] GPU:', utils.get_GPU_memory())
        return True

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

    @torch.no_grad()
    def _build_gt_segmentation(self, gt_tree: 'Tree3Dv2', tri_ids: Tensor, ignore_pixels=100):
        """build gt 2D segmenation results using ground truth tree"""
        self._masks_2d_packed = False
        self.V = tri_ids.shape[0]
        face_masks = []
        # gt_masks = []
        view_indices = []
        num_masks = []
        self.range_view = []
        self.view_infos = []
        self.masks_view = torch.zeros((self.V, gt_tree.num_faces), dtype=torch.bool, device=self.device)
        temp = torch.zeros(gt_tree.num_faces, device=self.device)

        now_view = 0
        for vid in range(self.V):
            tri_id = tri_ids[vid].to(self.device)
            v_faces, v_cnts = self.get_face_list(tri_id)
            masks_2d = []
            for i in range(gt_tree.cnt):
                mask = gt_tree.masks[i, tri_id]
                if mask.sum().item() <= ignore_pixels:  # ignore the mask which the number of pixels less than threshold
                    continue
                faces, cnts = self.get_face_list(tri_id, mask)
                masks_2d.append(mask)
                temp.zero_()
                temp[faces.long()] = cnts.float()
                # faces_masks_v.append(temp[v_faces] / v_cnts)
                temp[v_faces] /= v_cnts
                face_masks.append(temp.clone())
            num_masks_vid = len(masks_2d)
            if num_masks_vid == 0:
                continue
            self.view_infos.append((v_faces, v_cnts))
            self.masks_view[vid, v_faces] = 1  # v_cnts.float()
            num_masks.append(num_masks_vid)
            # gt_masks.append(torch.stack(masks_2d, dim=0).cpu())
            self.range_view.append((len(view_indices), len(view_indices) + num_masks_vid))
            view_indices.extend([now_view] * num_masks_vid)
            now_view += 1
        if len(face_masks) == 0:
            print(f"Can not load gt, due there is no masks")
            return False
        self.masks_2d = torch.stack(face_masks, dim=0)
        self._masks_2d_sp = None
        self.indices_view = torch.tensor(view_indices, dtype=torch.int32, device=self.device)
        # print(tree3d_gt.view_range, utils.show_shape(tree3d_gt.face_masks))
        self.M = sum(x for x in num_masks)
        self.Lmax = max(x for x in num_masks)
        self.face_mask = F.pad(self.masks_view.any(0), (1, 0))
        print(f'[Tree3D] view_masks, view_infos[0]: {utils.show_shape(self.masks_view, self.view_infos[0])}')
        print(f"[Tree3D] loaded {self.V} views, {self.M} masks, max_num: {self.Lmax}")
        return True

    def reverse_masks_2d_pack(self, device=None, threshold=0.9):
        if device is None:
            device = self.device
        if self._masks_2d_packed:
            self._masks_2d_sp = None
            self.indices_2d = None
            masks_2d = torch.zeros((self.M, self.num_faces), dtype=torch.float, device=device)
            # indices = torch.arange(self.num_faces + 1, device=device)
            # for i in range(len(self.masks_2d)):
            #     masks_2d[self.masks_2d[i].to(device), indices] = 1
            # masks_2d = masks_2d[1:]
            print(self.masks_2d.shape, masks_2d.shape, len(self.range_level))
            for i in range(len(self.masks_2d)):
                for j in range(*self.range_2d[i]):
                    masks_2d[j] = self.masks_2d[i] == (j + 1)
            self.masks_2d = None
            self.range_level = []
            self.range_2d = []
            self.masks_2d = masks_2d.to(self.device)
            self._masks_2d_packed = False
        else:
            self._masks_2d_sp = self.masks_2d.to_sparse_coo()
            masks_2d = []
            self.indices_2d = torch.zeros(self.M, dtype=torch.int, device=self.device)
            self.range_level = []
            self.range_2d = []
            for v in range(self.V):
                sl = len(masks_2d)
                masks_2d.append(torch.zeros(self.num_faces, dtype=torch.int, device=device))
                s, e = self.range_view[v]
                self.range_2d.append((s, s + 1))
                for i in range(s, e):
                    mask = self.masks_2d[i] >= 0.5
                    num = mask.sum()
                    if num == 0 or ((mask & (masks_2d[-1] == 0)).sum() / num) >= threshold:
                        masks_2d[-1][mask] = i + 1
                        self.range_2d[-1] = (self.range_2d[-1][0], i + 1)
                    else:
                        masks_2d.append(mask.int() * (i + 1))
                        self.range_2d.append((i, i + 1))
                    self.indices_2d[i] = len(masks_2d) - 1
                self.range_level.append((sl, len(masks_2d)))
            self.masks_2d = torch.stack(masks_2d).to(self.device)
            self._masks_2d_packed = True

    def build_view_graph(self, threshold=0.5, num_nearest=5):
        if self.verbose > 0:
            print(f"[Tree3D] start build view graph")
        assert self.masks_view is not None
        area = torch.mv(self.masks_view.float(), self.area)
        # print(area.shape)
        A = F.linear(self.masks_view.float(), self.masks_view * self.area)
        # A = A / (area[:, None] + area[None, :] - A).clamp_min(1e-7)
        A = A / area[:, None]
        indices = torch.topk(A, num_nearest + 1, dim=0)[1]
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
                    area_now = self.area[view_mask]
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
                    inter = F.linear(masks_i_, masks_j * self.area)
                    area_i = (masks_i_ * self.area).sum(-1)
                    area_j = (masks_j * self.area).sum(-1)
                    iou = inter / (area_i[:, None] + area_j[None, :] - inter).clamp_min(1e-7)
                    A[si:ei, sj:ej] = iou
                    A[sj:ej, si:ei] = iou.T
        # self.A = A
        return A

    def build_all_graph(self, threshold=0.5, num_nearest=5):
        """build a adjacency matrix between view and view, view and masks, masks and masks"""
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
                areas_m[i + 1] = torch.sum(mask * self.area)
            temp = torch.zeros((self.M + 1), device=self.device)
            for i in range(self.V):
                masks_2d = self.masks_2d * self.masks_view[i]
                temp.zero_()
                # for j in range(len(masks_2d)):
                #     temp.scatter_reduce_(0, masks_2d[j].long(), self.area, 'sum')
                for j in range(self.M):
                    mask = masks_2d[self.indices_2d[j]] == (j + 1)
                    temp[j + 1] = torch.sum(mask * self.area)
                A[:M, M + i] = temp[1:] / areas_m[1:].clamp_min(1e-7)
        else:
            areas_m = torch.mv(self.masks_2d, self.area)[:, None].clamp_min(1e-7)
            A[:M, M:] = F.linear(self.masks_2d, self.masks_view.float() * self.area) / areas_m
            A[M:, :M] = A[:M, M:].T
        # mask-mask
        A[:M, :M] = self.build_graph()
        return A

    @torch.enable_grad()
    def compress_masks(self, hidden_dims=(256, 256, 256), epochs=10000, batch_size=64, lr=1e-3, include_views=True):
        autoencoder = AutoEncoder(self.num_faces, hidden_dims).to(self.device)
        metric = ext.DictMeter()
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
        return X, autoencoder

    def calc_asssign_score(self, P: Tensor, Masks: Tensor = None, eps=1e-7):
        """计算community和masks两两间的dice score"""
        mask = self.masks_view[self.indices_view]
        if Masks is None:
            Masks = (P.T @ self.masks_2d) / (P.T @ mask).clamp_min(eps)  # shape: [C, num_faces]
        inter = F.linear(Masks, self.masks_2d * self.area)  # shape: [C, N], do not need *mask
        # now_area = (Masks[:32, None, 1:] * mask[None, :64, 1:] * mesh_area).sum(dim=-1)
        now_area = F.linear(Masks, mask * self.area)
        # print(inter.shape, now_area.shape, now_area_2.shape, (now_area_2 - now_area).abs().max())
        mask_areas = (self.masks_2d * self.area).sum(dim=-1)

        dice_loss = 2. * inter / (now_area + mask_areas[None, :]).clamp_min(eps)
        return dice_loss

    # mask应与对应社区相似
    def loss_comm(self, P: Tensor, scores: Tensor = None, Masks: Tensor = None, eps=1e-7):
        if scores is None:
            scores = self.calc_asssign_score(P, Masks, eps)
        return 1. - (scores * P.T).sum(dim=0).mean()

    def loss_2d(self, P: Tensor, eps=1e-7):
        """
        1. 每个mask选择社区
        2. 计算每个社区对应的faces
        3. 最大化社区与每个视角结果间的关系: P(view_i | C)
        """
        c_idx = torch.argmax(P, dim=1)
        c_masks = self.masks_2d.new_zeros((P.shape[1], self.num_faces + 1))
        c_masks = torch.scatter_add(c_masks, 0, c_idx, self.masks_2d)
        t = torch.scatter_add(torch.zeros(c_masks), 0, c_idx, self.masks_view[self.indices_view])
        c_masks = c_masks / t.clamp_min(1e-7)  # shape: [C, nF]
        c_masks = c_masks[c_idx]  # shape: [N, nF]

        inter = (c_masks * self.masks_2d * self.area).sum(dim=-1)  # shape: [N], do not need *mask
        # now_area = (Masks[:32, None, 1:] * mask[None, :64, 1:] * mesh_area).sum(dim=-1)
        now_area = (c_masks[:, 1:] * self.masks_view[self.indices_view] * self.area).sum(dim=-1)
        mask_areas = (self.masks_2d * self.area).sum(dim=-1)
        dice_loss = 2. * inter / (now_area + mask_areas[None, :]).clamp_min(eps)  # shape: [C, N]
        return

    def calc_comm_prob(self, P: Tensor):
        log_P = torch.log(1 - P)
        view_P = torch.zeros((self.V, P.shape[1]), device=P.device)  # shape: [V, C]
        view_P = torch.scatter_reduce(view_P, 0, self.indices_view[:, None].expand_as(log_P), log_P, 'sum')
        view_P = 1 - torch.exp(view_P)  # shape: [V, C]
        return view_P

    # def calc_loss_2(self, P: Tensor, feats: Tensor = None, eps=1e-7):
    #     if feats is None:
    #         feats = P.T @ mask_feats
    #     masked_feats = feats[:, None] * view_feats[self.view_indices][None, :]
    #     return 1 - F.cosine_similarity(masked_feats, mask_feats[None, :], dim=-1).mean()

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

    def loss_masks_relation(self, P: Tensor, A: Tensor):
        """ All 2D masks belonging to the same 3D masks should be connected to each other or cannot see each other
        m_a, m_b -> M_k,  A[a, b] = 1 or V[a, b] = 0
      
        Args:
            P: shape [M, K]
            A: shape [M+V, M+V]
        """
        in_same = F.linear(P, P)  # P @ P.T # shape [M, M]
        in_same * torch.maximum(A[:self.M, :self.M],)

    def loss_masks_connection(self, P: Tensor, A: Tensor):
        """ All 2D masks belonging to the same 3D masks should be connected to each other, 
            ie, m_a, m_b -> M_k, exist a path (a, ..., b) in graph
        Args:
            P: shape [M, K]
            A: shape [M+V, M+V]
        """
        for k in range(P.shape[1]):
            for i in range(self.M):
                for j in range(self.M):
                    P[i, k] * P[j, k] - A[i, j]

    def loss_edge_similarity(self, S: Tensor, A: Tensor):
        S = F.normalize(S, dim=1)  # shape: [K, M]
        sim = S.T @ S  # shape: [M, M]
        return F.mse_loss(sim, A[:self.M, :self.M])

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
        area_v = self.area[view_mask]
        inter = F.linear(masks_v, masks_v * area_v)
        area = torch.mv(masks_v, area_v)
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
            match_score: shape [K, M_v]
            scores: shape [K]
        """
        idx3d, idx2d = linear_sum_assignment((1 - match_scores).detach().cpu().numpy())  # 二分匹配
        idx3d, idx2d = utils.tensor_to(idx3d, idx2d, device=match_scores.device)
        score_vk = match_scores[idx3d, idx2d]
        return 1 - 2 * (score_vk * scores[idx3d]).sum() / (scores[idx3d].sum() + match_scores.size(1))

    def loss_mask_view(self, match_scores: Tensor, scores: Tensor):
        """ Given a view, each 3D masks etheir match a 2D mask or empty or not in results
        Args:
            match_score: shape [K, M_v]
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

    def loss_mask_all_view(self, masks: Tensor):
        """ For a 3D mask, all view have a 2D match or empty

        Args:
            masks: shape [K, F+1]
        """

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

        f_area = self.area[self.masks_view[view]]  # [Fv]
        areas = torch.mv(masks, f_area)  # [K]
        inter = F.linear(masks, masks * f_area)
        In = inter / areas[:, None].clamp_min(eps)  # [K, K]
        IoU = inter / (areas[:, None] + areas[None, :] - inter).clamp_min(eps)  # [K, K]
        notIn = (P.T @ (1 - In) @ P) * R[0]  # [Mv, Mv]
        IoU = (P.T @ IoU @ P) * R[1]  # [Mv, Mv]
        loss = (notIn.sum() + IoU.sum()) / R.sum().clamp_min(eps)
        return loss

    @torch.no_grad()
    def _build_tree_for_loss(self, masks: Tensor):
        K = len(masks)
        areas = torch.mv(masks, self.area)

        def _build_tree(x: int, t: TreeStructure, p=0, threshold=0.5):
            in_set = []
            for c in t.get_children(p):
                mask_c = masks[c - 1]
                inter = torch.sum(masks[x - 1] * mask_c * self.area)
                in_i = inter / areas[x - 1].clamp_min(1e-7)
                in_c = inter / areas[c - 1].clamp_min(1e-7)
                if max(in_i, in_c) >= threshold:
                    if in_i > in_c:  # x in c
                        _build_tree(x, t, c, threshold)
                        return
                    else:  # c in x
                        in_set.append(c)
            t.node_insert(x, p)
            for c in in_set:
                t.node_move(c, x)
            return

        tree = TreeStructure(K + 1, device=self.device)
        # print(tree.cnt, masks.shape)
        for i in range(K):
            idx = tree.node_new()
            assert idx == i + 1
            _build_tree(idx, tree)
        return tree

    def loss_tree_2(self, masks: Tensor, scores: Tensor, eps=1e-7):
        t = self._build_tree_for_loss(masks)
        R = self._get_node_relationship(t)

        ## check
        def is_parent(x, y):
            x, y = x + 1, y + 1
            if x == y:
                return False
            while x != 0:
                if x == y:
                    return True
                x = t.parent[x].item()
            return False

        # for i in range(K):
        #     for j in range(K):
        #         if i == j:
        #             assert not R[0,i, j] and not v_c[i, j]
        #         else:
        #             assert R[0, i, j] == is_parent(i, j), f"{i} {j}, {R[0, i, j]}, {is_parent(i, j)}"
        #             assert v_c[i, j] != (is_parent(i, j) or is_parent(j, i)), \
        #             f"{i} {j}, {v_c[i, j]}, {is_parent(i, j)},  {is_parent(j, i)}"
        ## calc loss
        conflict = (1 - masks[None, :, :]) * R[0, :, :, None] + masks[None, :, :] * R[1, :, :, None]
        conflict = conflict * masks[:, None, :] * scores[None, :, None]
        conflict = conflict.amax(dim=1)
        ## check
        # for i in range(K):
        #     cf_i = torch.zeros_like(masks[i])
        #     for j in range(K):
        #         if v_p[i, j]:
        #             cf_i = torch.maximum(cf_i, (1 - masks[j]) * masks[i] * scores[j])
        #         if v_c[i, j]:
        #             cf_i = torch.maximum(cf_i, masks[j] * masks[i] * scores[j])
        #     assert (conflict[i] - cf_i).abs().max() < 1e-5, f"{(conflict[i]-cf_i).abs().max()}"
        loss = torch.mv(conflict, self.area) / torch.mv(masks, self.area).clamp_min(eps)
        loss = (loss * scores).mean()
        return loss

    def loss_tree(self, masks: Tensor, scores: Tensor, eps=1e-7):
        """let masks to be a tree"""
        areas = masks @ self.area
        inter = F.linear(masks, masks * self.area)
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

    def _get_masks(self, P: Tensor, eps=1e-7):
        # P shape: [K, M]
        assert 0 <= self.indices_view.min() and self.indices_view.max() < self.V
        torch.cuda.synchronize()
        weights = scatter(P, self.indices_view.long(), dim=1, dim_size=self.V, reduce='sum')
        torch.cuda.synchronize()
        weights = (weights @ self.masks_view.float()).clamp_min(eps)
        assert P.shape[1] == self.M
        if self._masks_2d_packed:
            masks = (P @ self._masks_2d_sp) / weights
        else:
            masks = (P @ self.masks_2d) / weights
        return masks  # shape: [K, F]

    def calc_losses(
            self,
            logits: Tensor,
            node_logits: Tensor,
            view_index: int,
            A: Tensor,
            eps=1e-7,
            progress=1.0,
            timer: utils.TimeWatcher = None,
            weights=dict(),
    ) -> Dict[str, Tensor]:
        losses = {}  # type: Dict[str, Tensor]
        if progress < 0 or progress == 1.:
            P = logits.softmax(dim=1).T
        else:
            topP, indices = torch.topk(logits, k=max(1, int(logits.size(1) * (1 - progress))), dim=1)
            topP = topP.softmax(dim=1)
            P = torch.scatter(torch.zeros_like(logits), 1, indices, topP).T

        masks = self._get_masks(P)  # [K, F]
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
        area_3d = torch.mv(masks, view_mask * self.area)  # shape: [K]
        if self._masks_2d_packed:
            with torch.no_grad():
                masks_2d = []
                for i in range(s, e):
                    assert 0 <= self.indices_2d[i] and self.indices_2d[i] < len(self.masks_2d)
                    masks_2d.append(self.masks_2d[self.indices_2d[i]] == (i + 1))
                masks_2d = torch.stack(masks_2d, dim=0).float()
        else:
            assert 0 <= s and e <= len(self.masks_2d)
            masks_2d = self.masks_2d[s:e].float()
        inter = F.linear(masks, masks_2d * self.area)  # shape: [K, N], do not need *view_mask
        # print(inter.shape, now_area.shape, now_area_2.shape, (now_area_2 - now_area).abs().max())
        area_2d = torch.mv(masks_2d, self.area)  # shape: [Nv] 可以预处理, do not need *view_mask
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

            seen = area_3d / torch.mv(masks, self.area).clamp_min(eps)
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

        if weights.get('tree2', 0) > 0:
            losses['tree'] = self.loss_tree_2(masks, scores)
            if timer is not None:
                timer.log('tree2 loss')
        # assert not any(torch.isnan(x) for x in losses.values()), losses
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
        torch.set_anomaly_enabled(True)
        # print('[Tree3D] GPU:', utils.get_GPU_memory())
        # if self.view_masks is None or (N_view > 0 and N_view != self.N_view):
        #     self.load_2d_results(self.save_root, N_view)
        assert self.masks_view is not None
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
        meter = ext.DictMeter()
        timer = utils.TimeWatcher()
        if weights is None:
            weights = {}
        weights = {**dict(match=1, view=1, mv=1, recon=1, t2d=1, tree=1, vm=1), **weights}
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
        self.masks_area = torch.mv(self.masks[:, 1:].float(), self.area)
        self.scores *= self.masks_area > 0  # remove empty mask
        self.cnt = 0
        self.first[0] = -1
        self.resize(K + 1)
        for i in range(self.masks.shape[0]):
            index = self.node_new()
            if self.scores[i] >= self.score_threshold:
                self.insert(index, 0)

    def set_score_threshold(self, threshold):
        print(f"[Tree3D] set score theshold={threshold}")
        self.score_threshold = threshold
        if self.masks is None:
            return
        self.cnt = 0
        self.parent[:] = -1
        self.last[:] = -1
        self.next[:] = -1
        self.first[:] = -1
        if self.masks_area is None:
            self.masks_area = torch.mv(self.masks[:, 1:].float(), self.area)
        for i in range(self.masks.shape[0]):
            index = self.node_new()
            if self.scores[i] >= self.score_threshold:
                self.insert(index, 0)

    def save(self, save_path):
        torch.save({name: getattr(self, name, None) for name in self._save_list}, save_path)
        print('[Tree3D] Save results to', save_path)

    def load(self, save_path):
        for name, value in torch.load(save_path, map_location=self.device).items():
            setattr(self, name, value)
        print('[Tree3D] load results from:', save_path)
        # self.print_tree()

    def get_aux_data(self, tri_id: Tensor):
        aux_data = {}
        for nodes in reversed(self.get_levels()):
            for x in nodes:
                x = x.item()
                if x == 0:
                    continue
                mask = self.masks[x - 1][tri_id]
                area = mask.sum().item()
                if area > 0:  # self.ignore_area:
                    aux_data[x] = (mask, area)
        mask = tri_id > 0
        aux_data[0] = (mask, mask.sum().item())
        aux_data['tri_id'] = tri_id
        # print('[Tree3D] get_aux_data:', utils.show_shape(aux_data))
        # print(f'There are {len(aux_data) - 1} segmented masks')
        return aux_data

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

    def to(self, device):
        super().to(device)
        self.scores = self.scores.to(device)
        self.area = self.area.to(device)
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

    @classmethod
    def convert(cls, other: Tree3D):
        new = cls(other.mesh, device=other.device, verbose=other.verbose)
        new.reset()
        new.resize(other.cnt)
        other.node_rearrange()
        new.parent = other.parent
        new.first = other.first
        new.next = other.next
        new.last = other.last
        new.cnt = other.cnt
        new.area = other.area
        new.masks = torch.zeros((other.cnt, other.num_faces + 1), device=other.device, dtype=torch.bool)
        new.scores = torch.ones((other.cnt,), device=other.device)

        temp = other.face_parent.clone()
        for nodes in reversed(other.get_levels()):
            for x in nodes:
                x = x.item()
                if x == 0:
                    continue
                new.masks[x - 1] = temp == x
                temp[new.masks[x - 1]] = other.parent[x].to(temp.dtype)
        return new

    def get_2d_tree(self, tri_id: Tensor, ignore=100):
        aux_data = self.get_aux_data(tri_id)
        levels = self.get_levels(aux_data)
        masks = []
        scores = []
        for level in levels[1:]:
            for x in level:
                mask, area = aux_data[x.item()]
                if area > ignore:
                    masks.append(mask)
                    scores.append(self.scores[x - 1])
        if len(masks) == 0:
            masks = torch.zeros((0, *tri_id.shape), dtype=torch.bool, device=tri_id.device)
            scores = torch.zeros((0,), dtype=torch.float, device=tri_id.device)
            return Tree2D(masks, scores, device=tri_id.device)
        tree2d = Tree2D(torch.stack(masks), torch.stack(scores), device=tri_id.device)
        tree2d.update_tree()
        tree2d.node_rearrange()
        tree2d.post_process()
        return tree2d


class AutoEncoder(nn.Module):

    def __init__(self, num_faces: int, hidden_dims: Sequence[int] = (256, 256), use_bn=True):
        super().__init__()
        self.num_layers = len(hidden_dims)
        layers = []
        hidden_dims = [num_faces] + list(hidden_dims)
        for i in range(self.num_layers):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if i != self.num_layers - 1:
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
                layers.append(nn.GELU())
        self.encoder = nn.Sequential(*layers)
        layers = []
        for i in range(self.num_layers, 0, -1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i - 1]))
            if i > 1:
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dims[i - 1]))
                layers.append(nn.GELU())
        self.decoder = nn.Sequential(*layers)

    def forward(self, masks, only_encoder=False):
        features = self.encoder(masks)
        if only_encoder:
            return features
        return features, self.decoder(features)


# 相机间距离由重合的triangle的数量决定
