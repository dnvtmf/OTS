import torch
from torch import Tensor
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from extension import Mesh
from tree_segmentation.tree_2d_segmentation import TreeStructure


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
        # self.score_node = torch.empty((1, 2), device=device, dtype=torch.int)
        super().__init__(1, device=device, verbose=verbose)
        self.total_score = 0
        self.nodes_info = {}
        # merge parameters
        self.threshold_in = in_threshold
        self.threshold_in_area = in_thre_area
        self.threshold_union = union_threshold
        self.threshold_score = 0.5
        self.threshold_score_node = 0.5
        self.momentum = momentum
        self.ignore_area = ignore_area
        self.pad_length = 10  # add 10 empty nodes when enlarge nodes

    def reset(self):
        self.face_parent.fill_(-1)
        self.score.fill_(-1)
        # self.score_node.fill_(5.)
        self.total_score = 0
        self.nodes_info = {}
        super().reset()

    def resize(self, N: int):
        N, M = super().resize(N)
        if M > N:
            self.score = self.score[:N]
            # self.score_node = self.score_node[:N]
            self.nodes_info = {k: v for k, v in self.nodes_info.items() if k < N}
        elif M < N:
            self.score = torch.constant_pad_nd(self.score, (0, 0, 0, N - M), -1)
            # self.score_node = torch.constant_pad_nd(self.score_node, (0, 0, 0, N - M), 5.)
            # logger.NOTE(utils.show_shape(self.score, self.node_score))

    def node_new(self):
        if self.cnt + 1 == len(self.parent):
            self.resize(self.cnt + 1 + self.pad_length)
        index = super().node_new()
        self.score[index].fill_(-1)
        self.nodes_info[index] = []
        # self.score_node[index].fill_(5)
        return index

    def node_rearrange(self, indices=None):
        indices, new_indices = super().node_rearrange(indices)
        num = len(indices)
        self.face_parent = new_indices[self.face_parent + 1]
        self.score[:num] = self.score[indices]
        # self.score_node[:num] = self.score_node[indices]
        self.nodes_info = {new_indices[k + 1]: v for k, v in self.nodes_info.items()}
        return indices, new_indices

    def get_aux_data(self, tri_id: Tensor):
        # indices = self.face_parent[tri_id]
        aux_data = {}
        for nodes in reversed(self.get_levels()):
            for x in nodes:
                # mask = indices.eq(x)
                x = x.item()
                if x == 0:
                    continue
                mask = self.score[x] / len(self.nodes_info[x]) > self.threshold_score
                mask = mask[tri_id]
                area = mask.sum().item()
                if area >= self.ignore_area:
                    aux_data[x] = (mask, area)
                # indices[mask] = self.parent[x].item()
        mask = tri_id > 0
        aux_data[0] = (mask, mask.sum().item())
        aux_data['tri_id'] = tri_id
        aux_data['tri_uni'] = tri_id.unique(return_counts=True)
        # print('[Tree3D] get_aux_data:', utils.show_shape(aux_data))
        # print(f'There are {len(aux_data) - 1} segmented masks')
        return aux_data

    def get_level(self, aux_data: dict = None, root=0, depth=1, include_faces=False):
        results = self.get_levels(aux_data, root, depth, include_faces)
        return results[depth] if len(results) > depth else torch.tensor([])

    def get_levels(self, aux_data: dict = None, root=0, depth=-1, include_faces=False):
        levels = super().get_levels(root=root, depth=depth)
        # logger.WARN(f'[Tree3D] levels without auxdata:', levels)
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
        #     logger(f'[Tree3D] get {len(levels)} levels')
        return levels

    def add_one_mask(self, mask: Tensor, aux_data: dict, info):
        max_iou = (0, 0)
        node = 0
        area = mask.sum().item()
        while True:
            max_node = (0, 0)
            for child in self.get_children(node):
                mask_c, area_c = aux_data[child]
                inter = (mask & mask_c).sum().item()
                iou = inter / (area + area_c - inter)
                if iou > max_node[1]:
                    max_node = (child, iou)
            if max_node == (0, 0):
                break
            node = max_node[0]
            if max_node[1] > max_iou[1]:
                max_iou = max_node
        if True or max_iou[1] > self.threshold_in:
            tmp = torch.zeros(self.num_faces + 1, device=self.device)
            idx, cnt = aux_data['tri_id'][mask].unique(return_counts=True)
            tmp[idx] = cnt.float()
            tmp[aux_data['tri_uni'][0]] /= aux_data['tri_uni'][1]
            tmp[0] = 0
            self.score[max_iou[0]] += tmp
            self.nodes_info[max_iou[0]].append(info)
        # else:
        #     pass

    def calc_total_score(self):
        score_node = 0
        for i in range(1, self.cnt + 1):
            score_node = self.score[i] > self.threshold_score * len(self.nodes_info)
        score_tree = 0
        score_struct = 0
