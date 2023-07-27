import math
import os
import random
from pathlib import Path
from typing import List, Tuple, Optional, Sequence

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from scipy.optimize import linear_sum_assignment
from tree_segmentation.extension import Mesh, utils
from tree_segmentation.extension import ops_3d
from tree_segmentation import TreeStructure, MaskData, extension as ext


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

    def proposal_camera_pose_cycle(
        self, num=1, radius_range=(2.5, 3.), elev_range=(0, 180), azim_range=(0, 360), device=None
    ):
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
        print('num_nodes:',
            sum(len(x) for x in self.get_levels()),
            'num conflict:',
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
        torch.save({
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
    _save_list = ['masks', 'scores', 'parent', 'last', 'next', 'first', 'cnt', 'A']

    def __init__(self, mesh: Mesh, device=None, verbose=0):
        self.device = device
        # mesh
        self.mesh = mesh
        v3 = mesh.v_pos[mesh.f_pos]  # shape: (F, 3, 3)
        self.area = torch.cross(v3[:, 0] - v3[:, 1], v3[:, 0] - v3[:, 2], dim=-1).norm(dim=-1) * 0.5
        assert torch.all(self.area.ge(0))
        self.num_faces = mesh.f_pos.shape[0]
        # tree results
        self.masks = None
        self.scores = None
        super().__init__(1, device=device, verbose=verbose)
        self.ignore_area = 10
        self.score_threshold = 0.5

    def reset(self):
        super().reset()
        ## data
        self.view_infos = []  # type: List[Tuple[Tensor, Tensor]]
        self.view_masks = None  # type: Optional[Tensor]
        self.view_range = []  # type: List[Tuple[int, int]]
        self.view_indices = None  # type: Optional[Tensor]
        # self.faces_masks = []  # type: List[Tensor]
        self.face_masks = None  # type: Optional[Tensor]
        self.M = 0  # the total number of masks for all views
        # self.A = None
        ## results
        self.masks = None
        self.scores = None
        torch.cuda.empty_cache()

    @torch.no_grad()
    def load_2d_results(self, save_root: Path, N_view=-1):
        print('[Tree3D] GPU:', utils.get_GPU_memory())
        seg_2d_files = sorted(list(save_root.glob('*.data')))  # [:50]
        print(f'[Tree3D] There are {len(seg_2d_files)} data')
        if N_view > 0:
            N_view = min(N_view, len(seg_2d_files))
            print(f"[Tree3D] only load segmentation results of {N_view} view")
            seg_2d_files = seg_2d_files[:N_view]
        data = [torch.load(filename, map_location='cpu') for filename in seg_2d_files]  # type: List[dict]
        # self.data = data
        self.N_view = len(data)
        print('[Tree3D]', utils.show_shape(data[0]))
        print('[Tree3D] GPU:', utils.get_GPU_memory())

        face_masks = []
        num_masks = []
        view_indices = []
        self.view_range = []
        self.view_infos = []
        self.view_masks = torch.zeros((self.N_view, self.num_faces + 1), dtype=torch.float, device=self.device)
        temp = torch.zeros(self.num_faces + 1, device=self.device)
        for vid in range(self.N_view):
            masks = data[vid]['tree_data']['masks'].to(self.device)
            tri_id = data[vid]['tri_id'].to(self.device)

            v_faces, v_cnts = tri_id.unique(return_counts=True)
            if v_faces[0] == 0:
                v_faces, v_cnts = v_faces[1:], v_cnts[1:]
            self.view_infos.append((v_faces, v_cnts))
            self.view_masks[vid, v_faces] = 1  # cnts.float()

            faces_masks_v = []
            for i in range(len(masks)):
                if i == 0:  # skip foreground and background
                    continue
                for j in masks[i].unique():
                    j = j.item()
                    if j == 0:
                        continue
                    mask = masks[i] == j
                    faces, cnts = torch.unique(tri_id[mask], return_counts=True)
                    temp.zero_()
                    temp[faces.long()] = cnts.float()
                    # faces_masks_v.append(temp[v_faces] / v_cnts)
                    temp[v_faces] /= v_cnts
                    temp[0] = 0
                    faces_masks_v.append(temp.clone())

            num_masks.append(len(faces_masks_v))
            self.view_range.append((len(view_indices), len(view_indices) + num_masks[-1]))
            view_indices.extend([vid] * num_masks[-1])
            face_masks.append(torch.stack(faces_masks_v, dim=0))
        # self.face_masks = TensorSequence(*face_masks)
        self.face_masks = torch.cat(face_masks, dim=0)
        self.view_indices = torch.tensor(view_indices, dtype=torch.int32, device=self.device)
        # print(self.view_range, utils.show_shape(self.face_masks))

        self.M = sum(num_masks)
        self.Lmax = max(num_masks)
        print(f'[Tree3D] view_masks, view_infos[0]: {utils.show_shape(self.view_masks, self.view_infos[0])}')
        print(f"[Tree3D] loaded {self.N_view} views, {self.M} masks, max_num: {self.Lmax}")
        # print(utils.show_shape(self.face_masks.tensors, self.face_masks.index))
        print('[Tree3D] GPU:', utils.get_GPU_memory())

    @torch.no_grad()
    def build_gt_segmentation(self, gt_tree: 'Tree3Dv2', tri_ids: Tensor, ignore_pixels=100):
        """build gt 2D segmenation results using ground truth tree"""
        self.N_view = tri_ids.shape[0]
        face_masks = []
        # gt_masks = []
        view_indices = []
        num_masks = []
        self.view_range = []
        self.view_infos = []
        self.view_masks = torch.zeros((self.N_view, gt_tree.num_faces + 1), dtype=torch.float, device=self.device)
        temp = torch.zeros(gt_tree.num_faces + 1, device=self.device)

        for vid in range(self.N_view):
            tri_id = tri_ids[vid].to(self.device)
            v_faces, v_cnts = tri_id.unique(return_counts=True)
            if v_faces[0] == 0:
                v_faces, v_cnts = v_faces[1:], v_cnts[1:]
            self.view_infos.append((v_faces, v_cnts))
            self.view_masks[vid, v_faces] = 1  # v_cnts.float()
            masks_2d = []
            for i in range(gt_tree.cnt):
                mask = gt_tree.masks[i, tri_id]
                if mask.sum().item() <= ignore_pixels:  # ignore the mask which the number of pixels less than threshold
                    continue
                faces, cnts = torch.unique(tri_id[mask], return_counts=True)
                masks_2d.append(mask)
                temp.zero_()
                temp[faces.long()] = cnts.float()
                # faces_masks_v.append(temp[v_faces] / v_cnts)
                temp[v_faces] /= v_cnts
                temp[0] = 0
                face_masks.append(temp.clone())
            num_masks_vid = len(masks_2d)
            num_masks.append(num_masks_vid)
            # gt_masks.append(torch.stack(masks_2d, dim=0).cpu())
            self.view_range.append((len(view_indices), len(view_indices) + num_masks_vid))
            view_indices.extend([vid] * num_masks_vid)
        self.face_masks = torch.stack(face_masks, dim=0)
        self.view_indices = torch.tensor(view_indices, dtype=torch.int32, device=self.device)
        # print(tree3d_gt.view_range, utils.show_shape(tree3d_gt.face_masks))
        self.M = sum(x for x in num_masks)
        self.Lmax = max(x for x in num_masks)
        print(f'[Tree3D] view_masks, view_infos[0]: {utils.show_shape(self.view_masks, self.view_infos[0])}')
        print(f"[Tree3D] loaded {self.N_view} views, {self.M} masks, max_num: {self.Lmax}")

    def build_view_graph(self, threshold=0.5, num_nearest=5):
        if self.verbose > 0:
            print(f"[Tree3D] start build view graph")
        assert self.view_masks is not None
        area = torch.mv(self.view_masks[:, 1:], self.area)
        # print(area.shape)
        A = F.linear(self.view_masks[:, 1:], self.view_masks[:, 1:] * self.area)
        # A = A / (area[:, None] + area[None, :] - A).clamp_min(1e-7)
        A = A / area[:, None]
        indices = torch.topk(A, num_nearest + 1, dim=0)[1]
        # print(utils.show_shape(indices))
        A = A.ge(threshold)
        A[torch.arange(self.N_view), indices] = 1
        # print(A.sum(dim=1))
        return A

    def build_graph(self, view_graph: Tensor = None):
        if self.verbose > 0:
            print(f"[Tree3D] start build graph")
        # view_graph = self.build_view_graph()
        if view_graph is None:
            view_graph = torch.ones((self.N_view, self.N_view), device=self.device, dtype=torch.bool)
        N = view_graph.shape[0]
        M = self.face_masks.shape[0]
        A = torch.zeros((M, M), device=self.device, dtype=torch.float)
        for i in range(N):
            si, ei = self.view_range[i]
            for j in range(N):
                if i >= j or not view_graph[i, j]:
                    continue
                # print(f'i = {i}, j = {j}')
                # print(utils.show_shape(view_masks), torch.unique(view_masks))
                view_mask = (self.view_masks[i] * self.view_masks[j])[1:]
                sj, ej = self.view_range[j]
                assert torch.all(self.view_indices[si:ei] == i) and torch.all(self.view_indices[sj:ej] == j)  # noqa
                masks_i = self.face_masks[si:ei, 1:] * view_mask
                masks_j = self.face_masks[sj:ej, 1:] * view_mask
                inter = F.linear(masks_i, masks_j * self.area)
                area_i = (masks_i * self.area).sum(-1)
                area_j = (masks_j * self.area).sum(-1)
                # print(utils.show_shape(area_i, area_j, inter))
                iou = inter / (area_i[:, None] + area_j[None, :] - inter).clamp_min(1e-7)
                # print(iou)
                A[si:ei, sj:ej] = iou
                A[sj:ej, si:ei] = iou.T
        # self.A = A
        return A

    @torch.enable_grad()
    def get_graph_attr(self, hidden_dims=(256, 256, 256), epochs=10000, batch_size=64, lr=1e-3):
        autoencoder = AutoEncoder(self.num_faces, hidden_dims).to(self.device)
        metric = ext.DictMeter()
        opt = torch.optim.Adam(autoencoder.parameters(), lr=lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, 1e-6)

        autoencoder.train()
        for epoch in range(epochs):
            losses = {}
            # threshold = torch.rand(1, device=device)
            # edges = torch.nonzero(A.gt(threshold))
            # assert len(edges) > 0
            # edges = edges[torch.randint(0, len(edges), (batch_size // 2,))]
            # edges = torch.cat([edges, torch.randint(0, N, (batch_size - edges.shape[0], 2), device=device)], dim=0)
            # gt = faces_masks[edges.view(-1), 1:]
            # print(utils.show_shape(gt), *gt.aminmax())
            masks_gt = self.face_masks[torch.randint(0, self.M, (batch_size,)), 1:]
            # print(utils.show_shape(masks_gt))
            features_, masks_pred = autoencoder(masks_gt)
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

            losses['recon'] = F.binary_cross_entropy_with_logits(masks_pred, masks_gt)
            metric.update(losses)
            opt.zero_grad()
            sum(losses.values()).backward()
            opt.step()
            lr_scheduler.step()
            if epoch % 100 == 0:
                print(f'[Tree3D] X epoch[{epoch:4d}], loss: {metric.average}, lr={lr_scheduler.get_last_lr()[0]:.3e}')
                metric.reset()
        autoencoder.eval()
        X = []
        with torch.no_grad():
            for face_masks_part in self.face_masks.split(batch_size * 2, dim=0):
                # print(utils.show_shape(face_masks_part))
                X.append(autoencoder(face_masks_part[:, 1:], only_encoder=True))
        X = torch.cat(X, dim=0)
        print('[Tree3D] Features of face masks:', utils.show_shape(X))
        return X, autoencoder

    def calc_asssign_score(self, P: Tensor, Masks: Tensor = None, eps=1e-7):
        """计算community和masks两两间的dice score"""
        mask = self.view_masks[self.view_indices]
        if Masks is None:
            Masks = (P.T @ self.face_masks) / (P.T @ mask).clamp_min(eps)  # shape: [C, num_faces]
        inter = F.linear(Masks[:, 1:], self.face_masks[:, 1:] * self.area)  # shape: [C, N], do not need *mask
        # now_area = (Masks[:32, None, 1:] * mask[None, :64, 1:] * mesh_area).sum(dim=-1)
        now_area = F.linear(Masks[:, 1:], mask[:, 1:] * self.area)
        # print(inter.shape, now_area.shape, now_area_2.shape, (now_area_2 - now_area).abs().max())
        mask_areas = (self.face_masks[:, 1:] * self.area).sum(dim=-1)

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
        c_masks = self.face_masks.new_zeros((P.shape[1], self.num_faces + 1))
        c_masks = torch.scatter_add(c_masks, 0, c_idx, self.face_masks)
        t = torch.scatter_add(torch.zeros(c_masks), 0, c_idx, self.view_masks[self.view_indices])
        c_masks = c_masks / t.clamp_min(1e-7)  # shape: [C, nF]
        c_masks = c_masks[c_idx]  # shape: [N, nF]

        inter = (c_masks[:, 1:] * self.face_masks[:, 1:] * self.area).sum(dim=-1)  # shape: [N], do not need *mask
        # now_area = (Masks[:32, None, 1:] * mask[None, :64, 1:] * mesh_area).sum(dim=-1)
        now_area = (c_masks[:, 1:][:, 1:] * self.view_masks[self.view_indices, 1:] * self.area).sum(dim=-1)
        mask_areas = (self.face_masks[:, 1:] * self.area).sum(dim=-1)
        dice_loss = 2. * inter / (now_area + mask_areas[None, :]).clamp_min(eps)  # shape: [C, N]
        return

    def calc_comm_prob(self, P: Tensor):
        log_P = torch.log(1 - P)
        view_P = torch.zeros((self.N_view, P.shape[1]), device=P.device)  # shape: [V, C]
        view_P = torch.scatter_reduce(view_P, 0, self.view_indices[:, None].expand_as(log_P), log_P, 'sum')
        view_P = 1 - torch.exp(view_P)  # shape: [V, C]
        return view_P

    # def calc_loss_2(self, P: Tensor, feats: Tensor = None, eps=1e-7):
    #     if feats is None:
    #         feats = P.T @ mask_feats
    #     masked_feats = feats[:, None] * view_feats[self.view_indices][None, :]
    #     return 1 - F.cosine_similarity(masked_feats, mask_feats[None, :], dim=-1).mean()

    def loss_reg_edge_in_same_view(self, P: Tensor):
        mask = (self.view_indices[:, None] == self.view_indices[None, :]) ^ torch.eye(self.M, device=self.device).bool()
        same_view_edges = mask.nonzero(as_tuple=True)
        P_uv = (P[same_view_edges[0], :] * P[same_view_edges[1], :]).sum(dim=1)
        return P_uv.mean()

    # 相连的边的特征相似
    def loss_edge_similarity(self, S: Tensor):
        S = F.normalize(S, dim=1)  # shape: [N, C]
        sim = S @ S.T
        return F.mse_loss(sim, self.A)

    def loss_recon(self, P: Tensor, Masks: Tensor = None, k1=-1, k2=-1, eps=1e-7):
        if Masks is None:
            Masks = (P.T @ self.face_masks) / (P.T @ self.view_masks[self.view_indices]).clamp_min(eps)
        if k1 < 0:
            k1 = torch.randint(0, self.N_view, (1,)).item()
        if k2 < 0:
            k2 = torch.randint(0, self.N_view, (1,)).item()
        assert 0 <= k1 < self.N_view and 0 <= k2 < self.N_view
        view_mask = self.view_masks[k1].gt(0) * self.view_masks[k2].gt(0)
        view_mask[0] = 0
        # print('[Tree3D]', 'view_mask:', utils.show_shape(view_mask))
        if view_mask.sum() == 0:
            return torch.zeros(1, device=self.device)
        # print('[Tree3D]', *Masks.aminmax())
        masks_ = Masks[:, view_mask] * self.area[view_mask[1:]]
        # print('[Tree3D]', 'masks_', utils.show_shape(masks_), *mesh_area.aminmax())
        inter_ = F.linear(masks_, masks_)
        area = masks_.sum(dim=-1)
        # print('[Tree3D]', inter_.shape)
        IoU = inter_ / (area[:, None] + area[None, :] - inter_).clamp_min(eps)  # tree结点 两两间IoU
        # IoU = 2 * inter_ / (area[:, None] + area[None, :]).clamp_min(eps)  # tree结点 两两间IoU
        # print('[Tree3D]', IoU.shape)
        id_1 = torch.nonzero(self.view_indices.eq(k1))[:, 0]
        id_2 = torch.nonzero(self.view_indices.eq(k2))[:, 0]
        # print('[Tree3D]', utils.show_shape(id_1, id_2))
        predictions = (P[id_1, None, :, None] * P[None, id_2, None, :] * IoU).sum(dim=[2, 3])
        # print('[Tree3D]', (P[id_1, None, :, None] * P[None, id_2, None, :]).sum(dim=[2, 3]))
        # print('[Tree3D]', 'predictions', utils.show_shape(predictions))
        gt_index = torch.meshgrid(id_1, id_2, indexing='ij')
        # print('[Tree3D]', utils.show_shape(gt_index))
        IoU_gt = self.A[gt_index]
        # print('[Tree3D]', 'gt:', utils.show_shape(IoU_gt))
        # print('[Tree3D]', IoU_gt - predictions)
        return F.mse_loss(predictions, IoU_gt)

    def community_confidence(self, scores: Tensor, node_scores: Tensor):
        # print(scores.shape, *scores.aminmax())
        scores = 1. - scores
        losses = 0
        for i in range(self.N_view):
            scores_i = scores[:, self.view_indices == i]
            comm_idx, mask_idx = linear_sum_assignment(scores_i.detach().cpu().numpy())
            # print(comm_idx, mask_idx)
            comm_idx, mask_idx = utils.tensor_to(comm_idx, mask_idx, device=scores.device)
            score_ci = 1 - scores_i[comm_idx, mask_idx]
            # print(score_ci)
            dice_score = 2 * (score_ci * node_scores[comm_idx]).sum() / (node_scores.sum() + scores_i.shape[1])
            losses += 1 - dice_score
        return losses / self.N_view

    def loss_fn(self, S: Tensor, nodes_logits: Tensor, eps=1e-7):
        P = S.softmax(dim=1)  # shape: (N, C)
        Masks = (P.T @ self.face_masks) / (P.T @ self.view_masks[self.view_indices]).clamp_min(eps)  # [C, nF]
        match_score = self.calc_asssign_score(P, Masks, eps)
        node_scores = nodes_logits.sigmoid()
        losses = {
            'cm': self.loss_comm(P, scores=match_score, Masks=Masks, eps=eps),
            # 'edge': self.edge_similarity(S),
            # 'view': self.loss_reg_edge_in_same_view(P),
            # 'recon': self.recon_loss(P, Masks)
            # 'l1': self.calc_comm_prob(P).abs().mean(),
            'cc': self.community_confidence(match_score, node_scores),
        }
        return losses

    def calc_losses(self, logits: Tensor, node_logits: Tensor, view_index: int, eps=1e-7):
        s, e = self.view_range[view_index]
        # print(f'view_index: {view_index}, range: [{s}, {e})')
        P = logits.softmax(dim=1)
        # print(P.shape)

        Masks = (P.T @ self.face_masks) / (P.T @ self.view_masks[self.view_indices]).clamp_min(eps)  # [C, nF]
        mask = self.view_masks[view_index]  # 当前view的可见部分
        # 评估Masks投影到当前view后的masks与当前view检测出的结果之间的差别
        P = P[s:e]
        inter = F.linear(Masks[:, 1:], self.face_masks[s:e, 1:] * self.area)  # shape: [C, N], do not need *mask
        # now_area = (Masks[:32, None, 1:] * mask[None, :64, 1:] * mesh_area).sum(dim=-1)
        now_area = F.linear(Masks[:, 1:], mask[None, 1:] * self.area)  # 投影面积
        # print(inter.shape, now_area.shape, now_area_2.shape, (now_area_2 - now_area).abs().max())
        mask_areas = (self.face_masks[s:e, 1:] * self.area).sum(dim=-1)  # shape: [Nv] 可以预处理

        losses = {}
        match_score = 2. * inter / (now_area + mask_areas[None, :]).clamp_min(eps)  # dice score, shape: [C, Nv]
        # print('match_score:', utils.show_shape(match_score))
        losses['match'] = 1. - (match_score * P.T).sum(dim=0).mean()
        # losses['match'] = 1. - match_score[torch.argmax(P.T, dim=0), torch.arange(e - s, device=self.device)].mean()
        # 所有nodes与当前视角的匹配度
        node_scores = node_logits.sigmoid()  # the probability for where a node in the tree
        pred_idx, seg_idx = linear_sum_assignment((1 - match_score).detach().cpu().numpy())  # 二分匹配
        pred_idx, seg_idx = utils.tensor_to(pred_idx, seg_idx, device=match_score.device)
        score_ci = match_score[pred_idx, seg_idx]
        # print(score_ci)
        losses['mm'] = 1 - 2 * (score_ci * node_scores[pred_idx]).sum() / (
            node_scores[pred_idx].sum() + match_score.shape[1])  # dice loss
        return losses

    def run(self, epochs=10000, N_view=-1, K=0, gnn: nn.Module = None, A: Tensor = None, X: Tensor = None):
        torch.cuda.empty_cache()
        print('[Tree3D] GPU:', utils.get_GPU_memory())
        # if self.view_masks is None or (N_view > 0 and N_view != self.N_view):
        #     self.load_2d_results(self.save_root, N_view)
        assert self.view_masks is not None
        print('[Tree3D] GPU:', utils.get_GPU_memory())
        # if self.A is None:
        #     self.build_graph()
        K = 2 * self.Lmax if K <= 0 else K
        edge_weight = None
        if gnn is None:
            S = nn.Parameter(torch.randn((self.M, K), device=self.device))
            edges = None
        else:
            assert A is not None and X is not None
            assert A.shape == (self.M, self.M) and X.shape[0] == self.M
            edges = torch.nonzero(A).T
            if A.dtype.is_floating_point:
                edge_weight = A[edges[0], edges[1]]
            S = None
        node_score = nn.Parameter(torch.randn((K,), device=self.device))
        print('[Tree3D] GPU:', utils.get_GPU_memory())
        # nn.init.normal_(S)
        if gnn is None:
            opt = torch.optim.Adam([S, node_score], lr=1e-3)
        else:
            opt = torch.optim.Adam(list(gnn.parameters()) + [node_score], lr=1e-3)
        # opt = torch.optim.Adam(gnn.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, 1e-6)
        meter = ext.DictMeter()
        with torch.enable_grad():
            for epoch in range(epochs):
                view_index = random.randrange(self.N_view)
                # print(f"view_index: {view_index}")
                opt.zero_grad()
                if gnn is not None:
                    S = gnn(X, edges, edge_weight=edge_weight)
                # print('[Tree3D]', S.aminmax(), X.aminmax(), edge_weight.aminmax())
                # loss_dict = self.loss_fn(S, node_score)
                loss_dict = self.calc_losses(S, node_score, view_index)
                meter.update(loss_dict)
                total_loss = sum(loss_dict.values())
                # print('[Tree3D]', loss_dict)
                # break
                total_loss.backward()
                opt.step()
                lr_scheduler.step()
                if (epoch + 1) % 100 == 0:
                    print(f"[Tree3D] Epoch {epoch + 1}: loss={total_loss.item():.6f}, {meter.average}")
                    meter.reset()
                # break
        with torch.no_grad():
            scores = node_score.detach().sigmoid()
            self.scores, indices = torch.sort(scores, descending=True)
            if gnn is not None:
                gnn.eval()
                S = gnn(X, edges, edge_weight=edge_weight)
            P = S.softmax(dim=1)[:, indices]  # shape: (N, C)
            # self.scores = self.calc_comm_prob(P).mean(dim=0)
            Masks = (P.T @ self.face_masks) / (P.T @ self.view_masks[self.view_indices]).clamp_min(1e-7)  # [C, nF]
            self.masks = Masks >= 0.5
        self.cnt = 0
        self.first[0] = -1
        self.resize(K + 1)
        for i in range(self.masks.shape[0]):
            index = self.node_new()
            if self.scores[i] >= self.score_threshold:
                self.node_insert(index, 0)

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
        for i in range(self.masks.shape[0]):
            index = self.node_new()
            if self.scores[i] >= self.score_threshold:
                self.node_insert(index, 0)

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
