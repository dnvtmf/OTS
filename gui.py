import os

import yaml
import importlib
import math
from pathlib import Path
from time import time, sleep
from typing import Union
import gc
from rich.console import Console

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
import torch
from torch import Tensor

import tree_segmentation
import tree_segmentation.tree_2d_segmentation as ts2
import tree_segmentation.tree_3d_segmentation as ts3
from tree_segmentation.extension import utils
from tree_segmentation.extension import ops_3d
from tree_segmentation.extension.utils import ImageViewer, Viewer3D
from tree_segmentation import Tree3D, Tree3Dv2, MaskData, Tree2D
from tree_segmentation.tree_3d import TreeSegment
from tree_segmentation.util import color_mask, image_add_mask_boundary, image_add_points

console = Console()


def fit_window(mask: Tensor, region=0.95):
    """根据mask缩放图像, 使得所有点都恰好在框内"""
    """TODO: solve in 3D"""
    H, W = mask.shape
    assert region > 0
    points = torch.nonzero(mask).float() + 0.5
    center = points.new_tensor([0.5 * H, 0.5 * W])
    d = points - center
    left, right = 0.5 * W * (1. - region), 0.5 * W * (1. + region)
    t_left, t_right = (left - center[1]) / (d[:, 1] + 1e-10), (right - center[1]) / (d[:, 1] + 1e-10)
    top, bottom = 0.5 * H * (1. - region), 0.5 * H * (1. + region)
    t_top, t_bottom = (top - center[0]) / (d[:, 0] + 1e-10), (bottom - center[0]) / (d[:, 0] + 1e-10)
    t = torch.minimum(torch.maximum(t_left, t_right), torch.maximum(t_top, t_bottom))
    scale = 1. / t.min().item()
    return scale


class TreeSegmentGUI(TreeSegment):

    def __init__(self, model=None) -> None:
        super().__init__(None, model=model)
        self._mode = 'E2D'
        self._edit = False
        self._mesh_history = []  # The path of last 10 loaded meshes
        self.image_dir = Path('./images').expanduser()
        self.image_index = 0
        self.image_paths = []
        self._need_update_2d = False

        dpg.create_context()
        # self.set_default_font()
        dpg.create_viewport(title='Tree Segmentation', width=1024, height=1024, x_pos=800, y_pos=256, resizable=False)

        self.load_ini()
        self._now_level_2d = -1
        self._now_level_3d = 0
        self.last_time = time()
        self._loading_mesh = False
        self._choose_mask = 0
        self._edit_mask_idx = 0

        with dpg.theme() as self.choose_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Header, (150, 100, 100), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Button, (150, 100, 100), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core)
                # dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, (150, 100, 100), category=dpg.mvThemeCat_Core)
        with dpg.theme() as self.default_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (50, 50, 50), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Header, (50, 50, 50), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core)
                # dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, (50, 50, 50), category=dpg.mvThemeCat_Core)

        with dpg.window(tag='Primary Window', autosize=True):
            self.view_3d = Viewer3D(self.rendering, size=[512, 512], pos=(0, 0), no_resize=True, no_move=True)
            self.view_2d = ImageViewer(size=(512, 512), tag='image_sample', pos=(512, 0), no_resize=True, no_move=True)
            self.view_seg = ImageViewer(size=(512, 512), tag='image_seg', pos=(512, 512), no_resize=True, no_move=True)
            with dpg.window(
                    label='control',
                    tag='control',
                    width=512,
                    height=512 - 30,
                    pos=(0, 512),
                    no_close=True,
                    no_resize=True,
                    no_move=True,
                    # no_title_bar=True,
            ):
                # self.add_menu_bar()
                self.make_control_panel()
                # dpg.add_button(label='Exit', callback=exit)
            # with dpg.window(
            #     no_title_bar=True,
            #     width=512,
            #     pos=(0, 1024 - 30),
            #     autosize=True,
            #     no_resize=True,
            #     no_move=True,
            #     tag='win_status'
            # ):
            #     self.make_bottom_status_options()
            self.make_show_tree_2d_mask_level()
            self.make_show_tree_3d_mask_level()
            self.add_help_popup()
        dpg.set_primary_window('Primary Window', True)
        dpg.set_exit_callback(self.callback_exit)
        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(callback=self.view_3d.callback_mouse_drag)
            dpg.add_mouse_wheel_handler(callback=self.callback_mouse_wheel)
            dpg.add_mouse_release_handler(callback=self.view_3d.callback_mouse_release)
            dpg.add_mouse_move_handler(callback=self.callback_mouse_hover)
            dpg.add_mouse_click_handler(callback=self.callback_mouse_click)
            dpg.add_key_press_handler(callback=self.callback_keypress)

        self.view_3d.enable_dynamic_change()
        self.view_2d.enable_dynamic_change()
        self.view_seg.enable_dynamic_change()
        # dpg.set_primary_window('Primary Window', True)
        self.mode = self._mode

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        # if self._mode == mode:
        #     return
        print(f'[GUI] switch mode from {self._mode} to {mode}')
        self._tree_3d = None
        self.reset_2d()
        self.reset_3d()
        dpg.bind_item_theme(self._mode, self.default_theme)
        self._mode = mode
        dpg.bind_item_theme(self._mode, self.choose_theme)
        show_hidden_options = {
            self.view_2d.win_tag: [1, 0, 1],
            self.view_3d.win_tag: [0, 1, 1],
            self.view_seg.win_tag: [0, 0, 1],
            'ctl_predictor': [1, 1, 1],
            'ctl_edit_2d': [1, 0, 0],
            'ctl_edit_3d': [0, 1, 0],
            'ctl_view_3d': [0, 1, 1],
            'ctl_tree_3d': [0, 0, 1],
            'ctl_tree_2d': [0, 0, 1],
            'tree_0': [1, 1, 0],
            'copy_to_3d': [0, 0, 1],
            'copy_to_2d': [0, 0, 1],
            'get_uv': [0, 0, 1],
        }
        if mode == 'E2D':
            for item, values in show_hidden_options.items():
                if values[0]:
                    dpg.show_item(item)
                else:
                    dpg.hide_item(item)
            dpg.set_item_pos(self.view_2d.win_tag, [0, 0])
            # dpg.set_item_pos('control', [512, 0])
            self.change_image(self.image_index)
            dpg.configure_item('control', no_move=False)
            for i, level in enumerate(range(-1, 10)):
                dpg.move_item(f"level{level}", parent='Primary Window')
            self.change_image()
        elif mode == 'E3D':
            for item, values in show_hidden_options.items():
                if values[1]:
                    dpg.show_item(item)
                else:
                    dpg.hide_item(item)
            dpg.set_item_pos(self.view_3d.win_tag, [0, 0])
            dpg.configure_item('control', pos=[0, 512], no_move=True, height=512 - 30)
            # dpg.set_item_pos('win_status', [0, 1024 - 30])
            dpg.set_viewport_width(1024)
            dpg.set_viewport_height(1024)
        elif mode == 'S3D':
            for item, values in show_hidden_options.items():
                if values[2]:
                    dpg.show_item(item)
                else:
                    dpg.hide_item(item)
            dpg.set_item_pos(self.view_3d.win_tag, [0, 0])
            dpg.set_item_pos(self.view_seg.win_tag, [512, 512])
            dpg.configure_item(self.view_2d.win_tag, pos=[512, 0], width=512, height=512)
            self.view_2d.resize(512, 512, 3)
            self.view_2d.data[:] = 1
            self.view_2d._origin_data = None
            dpg.configure_item('control', pos=[0, 512], no_move=True, height=512 - 30)
            # dpg.set_item_pos('win_status', [0, 1024 - 30])
            dpg.set_viewport_width(1024)
            dpg.set_viewport_height(1024)
            for i, level in enumerate(range(-1, 10)):
                dpg.move_item(f"level{level}", parent=self.view_seg.win_tag)
                dpg.set_item_pos(f"level{level}", pos=[10 + (30 + 10) * i, 10])
        else:
            raise NotImplementedError()
        self.save_ini()

    @property
    def tree3d(self) -> Union[Tree3D, Tree3Dv2]:
        if self._tree_3d is None and self._mesh is not None:
            if self.mode == 'E3D':
                self._tree_3d = tree_segmentation.Tree3Dv2(self.mesh, device=self.device)
            else:
                self._tree_3d = tree_segmentation.Tree3D(self.mesh, device=self.device)
        return self._tree_3d

    @property
    def tree2d(self) -> Tree2D:
        if self._tree_2d is None:
            self._tree_2d = tree_segmentation.Tree2D(
                in_threshold=dpg.get_value('in_threshold'),
                in_thres_area=dpg.get_value('in_area_threshold'),
                union_threshold=dpg.get_value('union_threshold'),
                min_area=dpg.get_value('min_area'),
                device=self.device,
                verbose=1,
            )  # type: Tree2D
        if self._tree_2d._masks is None and self.tri_id is not None:
            background = self.tri_id == 0
            foreground = torch.logical_not(background)
            mask_data = MaskData(
                masks=torch.stack([foreground, background], dim=0),
                iou_preds=torch.ones(2, device=self.tri_id.device) * 2,
                # points=torch.zeros((2, 2), dtype=torch.float64),
                # stability_score=torch.ones(2, device=self.tri_id.device) * 2,
                # boxes=torch.zeros((2, 4), device=self.tri_id.device),
            )
            self._tree_2d.cat(mask_data)
            self._tree_2d.update_tree()
            self._tree_2d.num_samples[2] = background.numel()
        return self._tree_2d

    @property
    def image(self) -> np.ndarray:
        if self._image is None:
            self.new_camera_pose()
        return self._image

    @image.setter
    def image(self, img: np.ndarray):
        self._image = img
        self.view_2d.update(self._image, resize=True)

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points):
        self._points = points
        if points is None:
            self.view_2d.update(self.image, resize=True)
        else:
            self.view_2d.update(image_add_points(self.image, points, 5), resize=True)

    @property
    def mask_data(self):
        return self._mask_data

    @mask_data.setter
    def mask_data(self, data: MaskData = None):
        # if data is None:
        #     dpg.hide_item('level0')
        # else:
        #     dpg.show_item('level0')
        if data is not None:
            data.filter(torch.argsort(data['masks'].flatten(1).sum(dim=1), descending=True))
        self._mask_data = data

    @property
    def levels_2d(self):
        return self._2d_levels

    @levels_2d.setter
    def levels_2d(self, levels):
        max_levels = len(levels)
        for i in range(1, 10):
            dpg.configure_item(f"level{i}", show=i < max_levels)
            # dpg.focus_item(f"level{i}")
        self._2d_levels = levels
        self.show_tree_update()

    @property
    def now_level_2d(self):
        return self._now_level_2d

    @now_level_2d.setter
    def now_level_2d(self, level):
        if level == 0:
            self._points = None
            self._labels = None
            self._mask_data = None
        if self._now_level_2d >= -1:
            dpg.bind_item_theme(f"level{self._now_level_2d}", self.default_theme)
        self.choose_mask = 0
        self._now_level_2d = level
        if self._now_level_2d >= -1:
            dpg.bind_item_theme(f"level{self._now_level_2d}", self.choose_theme)
        self._need_update_2d = True

    @property
    def levels_3d(self):
        return self._3d_levels

    @levels_3d.setter
    def levels_3d(self, levels: list):
        max_levels = len(levels)
        for i in range(1, 10):
            dpg.configure_item(f"depth{i}", show=i < max_levels)
        self._3d_levels = levels
        if self.mode == 'E3D':
            if dpg.does_alias_exist(f"tree_{self.choose_mask}"):
                dpg.bind_item_theme(f"tree_{self.choose_mask}", self.choose_theme)

    @property
    def now_level_3d(self):
        return self._now_level_3d

    @now_level_3d.setter
    def now_level_3d(self, level=0):
        print(f'[GUI] 3D show level: {level}')
        dpg.bind_item_theme(f"depth{self._now_level_3d}", self.default_theme)
        self._now_level_3d = level
        dpg.bind_item_theme(f"depth{self._now_level_3d}", self.choose_theme)
        self.view_3d.need_update = True

    @property
    def choose_mask(self):
        return self._choose_mask

    @choose_mask.setter
    def choose_mask(self, index=0):
        # print('index:', index)
        if index == self._choose_mask:
            index = 0
        ch_tree = self._choose_mask > 0
        if self.mode == 'E2D':
            ch_tree = ch_tree and self.now_level_2d > 0
        # else:
        #     ch_tree = ch_tree and self.now_level_3d > 0
        if ch_tree and dpg.does_item_exist(f"tree_{self._choose_mask}"):
            dpg.bind_item_theme(f"tree_{self._choose_mask}", self.default_theme)
        self._choose_mask = index
        if ch_tree and dpg.does_item_exist(f"tree_{self._choose_mask}"):
            dpg.bind_item_theme(f"tree_{self._choose_mask}", self.choose_theme)

    def load_mesh(self, obj_path, use_cache=True, cache_suffix='.mesh_cache'):
        self._loading_mesh = True
        if self.get_value('mesh_load_all', False) and os.path.isfile(obj_path):
            obj_path = Path(obj_path).parent
        if dpg.get_value('mesh_load_force'):
            use_cache = 'force'
            dpg.set_value('mesh_load_force', False)
        if super().load_mesh(obj_path, use_cache, cache_suffix):
            obj_path = str(obj_path)
            if obj_path not in self._mesh_history:
                self._mesh_history.append(obj_path)
                if len(self._mesh_history) > 10:
                    self._mesh_history.pop(0)
            else:
                self._mesh_history.remove(obj_path)
                self._mesh_history.append(obj_path)
            for i, path in enumerate(reversed(self._mesh_history)):
                dpg.set_value(f"history_{i}", path)
            # dpg.configure_item('mesh_history_list', items=self._mesh_history)
            dpg.set_item_label('load_data', f"Load ({len(list(self.cache_dir.glob('*.data')))})")
            self.view_3d.need_update = True
        self._loading_mesh = False
        self.save_ini()

    def show_tree_update(self):
        # remove old tree nodes
        def _remove_children(name):
            for child in dpg.get_item_children(name, 1):
                _remove_children(child)
                dpg.delete_item(child)

        _remove_children(f"tree_0")
        if self.mode == 'S3D':
            return
        elif self.mode == 'E2D':
            if self._tree_2d is None:
                return
            tree = self.tree2d
        else:
            if self._tree_3d is None:
                return
            tree = self.tree3d
        # insert new tree nodes
        for level, nodes in enumerate(tree.get_levels()):
            if level == 0:
                continue
            # print(nodes)
            for x in nodes:
                x = x.item()
                p = tree.parent[x].item()
                dpg.add_collapsing_header(
                    label=f'{x}',
                    tag=f"tree_{x}",
                    parent=f"tree_{p}",
                    indent=level * 10,
                    leaf=tree.first[x].item() <= 0,
                    open_on_arrow=True,
                )
                dpg.bind_item_theme(f"tree_{x}", self.default_theme)
        dpg.configure_item('tree_0', default_open=True)
        print('[GUI]: update show tree')

    def get_value(self, name, default=None):
        if dpg.does_alias_exist(name):
            return dpg.get_value(name)
        elif name == 'fovy':
            return self.view_3d.fovy
        elif name == 'sam_path':
            return Path('./weights/sam_vit_h_4b8939.pth').expanduser()
        elif name == 'sam_l_path':
            return Path('./weights/sam_vit_l_0b3195.pth').expanduser()
        elif name == 'sam_b_path':
            return Path('./weights/sam_vit_b_01ec64.pth').expanduser()
        elif name == 'semantic_sam_l_path':
            return Path("./weights/swinl_only_sam_many2many.pth").expanduser()
        elif name == 'semantic_sam_t_path':
            return Path("./weights/swint_only_sam_many2many.pth").expanduser()
        else:
            print(f'[GUI] option {name} not in dpg')
            return default

    def _set_value(self, name, value):

        def setter(*args):
            return setattr(self, name, value)

        return setter

    def _cbw(self, func, *args, **kwargs):
        # callback
        def wrapper():
            return func(*args, **kwargs)

        return wrapper

    def _cbw2(self, cls, func, *args, update_3d=False, **kwargs):

        def wrapper():
            if isinstance(func, str):
                res = getattr(cls, func)(*args, **kwargs)
            else:
                f = cls
                for name in func:
                    f = getattr(f, name)
                res = f(*args, **kwargs)
            if update_3d:
                self.view_3d.need_update = True
            return res

        return wrapper

    @torch.no_grad()
    def rendering(self, Tw2v, fovy, size):
        if self._loading_mesh:
            return np.zeros((size[1], size[0], 3), dtype=np.float32)
        image, tri_id = self.render_mesh(Tw2v=Tw2v, image_size=size[0])
        image = image.cpu().numpy()
        if (time() - self.last_time) > 0.3:
            self._3d_aux_data = self.tree3d.get_aux_data(tri_id)
            self.levels_3d = self.tree3d.get_levels(self._3d_aux_data)
            # print('[3D] rendering', [x.tolist() for x in self.levels_3d])
            self.last_time = time()
        else:
            self.view_3d.need_update = True
            return image
        if 0 < self.now_level_3d < len(self.levels_3d):
            mask_index = self.levels_3d[self.now_level_3d]
            masks = torch.stack([self._3d_aux_data[x.item()][0] for x in mask_index], dim=0)
            self._3d_mask = masks * torch.arange(1, 1 + masks.shape[0], device=masks.device)[:, None, None]
            self._3d_mask = torch.amax(self._3d_mask, dim=0).int().cpu().numpy()
            self.view_3d._origin_data = None
            alpha = dpg.get_value('alpha')
            image = cv2.addWeighted(image, alpha, color_mask(self._3d_mask, masks.shape[0]), 1 - alpha, 0)
        if self.mode == 'E3D' and self.choose_mask > 0:
            if self.tree3d.masks is not None and 1 <= self.choose_mask <= len(self.tree3d.masks):
                mask = self.tree3d.masks[self.choose_mask - 1][tri_id]
                alpha = dpg.get_value('alpha')
                image = cv2.addWeighted(image, alpha, color_mask(mask.int().cpu().numpy(), 2), 1 - alpha, 0)
                image = image_add_mask_boundary(image, mask, color=(0, 1, 0))
        return image

    def show_2d(self):
        if self._image is None:
            return
        image = np.zeros_like(self.image)
        is_show_mask = False
        # show all masks
        masks = None
        if self.now_level_2d == 0:
            if self.mask_data is not None:
                masks = self.mask_data['masks']
        elif 0 < self.now_level_2d < len(self.levels_2d):
            mask_index = self.levels_2d[self.now_level_2d] - 1
            if mask_index.numel() > 0:
                assert 0 <= mask_index.min() and mask_index.max() < self.tree2d.num_masks
                masks = self.tree2d.masks[mask_index]

        if masks is not None and len(masks) > 0:
            self._2d_mask = masks * torch.arange(1, 1 + masks.shape[0], device=masks.device)[:, None, None]
            self._2d_mask = torch.amax(self._2d_mask, dim=0).int().cpu().numpy()
            image = color_mask(self._2d_mask, masks.shape[0])
            is_show_mask = True
        else:
            self._2d_mask = None
        # show_choose
        choose_mask = None
        if self.choose_mask > 0:
            if self.now_level_2d == 0:
                assert 0 <= self.choose_mask - 1 < len(self.mask_data['masks'])
                choose_mask = self.mask_data['masks'][self.choose_mask - 1]
            elif self.now_level_2d > 0 and self.choose_mask <= self.tree2d.num_masks:
                choose_mask = self.tree2d.masks[self.choose_mask - 1]

        if self.mode == 'S3D':
            if choose_mask is not None:
                image = image_add_mask_boundary(image, choose_mask, color=(0, 0., 0.))
            self.view_seg.update(image, resize=True)
        if is_show_mask:
            alpha = dpg.get_value('alpha')
            image = cv2.addWeighted(self.image, alpha, image[..., :3], 1 - alpha, 0)
        else:
            image = self.image
        if choose_mask is not None:
            image = image_add_mask_boundary(image, choose_mask, color=(0, 0., 0.))
        # show_edit mask
        if self._edit and self.mask_data is not None:
            mask = self.mask_data['masks'][int(self._edit_mask_idx) % len(self.mask_data['masks'])]
            image = image_add_mask_boundary(image, mask, color=(0, 0, 1.))
        # show_points
        if (self.now_level_2d == 0 or self._edit) and self._points is not None and self.get_value('show_points', True):
            pos_points = self._points if self._labels is None else self._points[self._labels]
            if len(pos_points) > 0:
                image = image_add_points(image, pos_points, s=3, color=(1., 0, 0))
            if self._labels is not None:
                neg_points = self._points[np.logical_not(self._labels)]
                if len(neg_points) > 0:
                    image = image_add_points(image, neg_points, s=3, color=(0, 1., 0))
        self.view_2d.update(image, resize=True)

    def run(self):
        dpg.setup_dearpygui()
        dpg.show_viewport()
        # dpg.set_primary_window('Primary Window', True)
        # dpg.start_dearpygui()
        last_size = None
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
            if self.mode != 'E2D':
                if self.view_3d.need_update:
                    dpg.set_value('eye_x', self.view_3d.eye[0].item())
                    dpg.set_value('eye_y', self.view_3d.eye[1].item())
                    dpg.set_value('eye_z', self.view_3d.eye[2].item())
                self.view_3d.update(False)
            if self._need_update_2d:
                self.show_2d()
                self._need_update_2d = False
            # now_size = dpg.get_item_width(view_3d._win_id), dpg.get_item_height(view_3d._win_id)
            # if last_size != now_size:
            #     dpg.configure_item('control', pos=(dpg.get_item_width(view_3d._win_id), 0))
            #     dpg.set_viewport_width(dpg.get_item_width(view_3d._win_id) + dpg.get_item_width('control'))
            #     dpg.set_viewport_height(dpg.get_item_height(view_3d._win_id))
            #     last_size = now_size
            dpg.set_value('fps', f"FPS: {dpg.get_frame_rate()}")

    def reset_2d(self):
        self.choose_mask = 0
        self.now_level_2d = -1
        super().reset_2d()
        if self._image is None:
            self.view_2d.data[:] = 255
        else:
            self.view_2d.update(self._image, True)
        self.view_seg.data[:] = 255

    def reset_3d(self):
        self.now_level_3d = 0
        super().reset_3d()
        self.show_tree_update()
        self.view_3d.need_update = True

    def reload(self):
        global tree_segmentation, ts2, ts3
        self.reset_2d()
        self.reset_3d()
        self._predictor = None
        self._tree_2d = None
        self._tree_3d = None
        tree_segmentation = importlib.reload(tree_segmentation)
        ts2 = importlib.reload(ts2)
        ts3 = importlib.reload(ts3)
        gc.collect()
        print('Reload successful!')

    def run_tree_seg_2d_stage1(self):
        self.mask_data = None
        self.now_level_2d = 0
        image = self.image
        points = self.tree2d.sample_grid(dpg.get_value('points_per_side'))
        # filter points in background
        # print(points.shape, points.min(), points.max(), points, self.tri_id.shape)
        if self.mode != 'E2D':
            x = np.clip(np.rint(points[:, 0] * self.tri_id.shape[1]), 0, self.tri_id.shape[1] - 1).astype(np.int32)
            y = np.clip(np.rint(points[:, 1] * self.tri_id.shape[0]), 0, self.tri_id.shape[0] - 1).astype(np.int32)
            points = points[self.tri_id.cpu().numpy()[y, x] > 0].reshape(-1, 2)
        self._points = points
        self._labels = np.ones_like(points[..., 0], dtype=bool)
        self.view_2d.update(image_add_points(image, points, 3), resize=True)
        if not self.predictor.is_image_set:
            self.predictor.set_image(np.clip(self.image * 255, 0, 255).astype(np.uint8))
        self.mask_data = self.predictor.process_points(points)
        self.tree2d.insert_batch(self.mask_data)
        # self.tree2d.update_tree()
        # self.tree2d.remove_not_in_tree()
        self.levels_2d = self.tree2d.get_levels()
        self.show_tree_update()

    def run_tree_seg_2d_stage2(self):
        self.mask_data = None
        self.now_level_2d = 0
        points = self.tree2d.sample_by_counts(dpg.get_value('points_per_update'))
        self._points = points
        self._labels = None if points is None else np.ones_like(points[..., 0], dtype=bool)
        if points is None:
            self.view_2d.update(self.image, resize=True)
            # print('ERROR: no unfilled mask')
            print(f'[GUI] Update complete')
            return False
        self.view_2d.update(image_add_points(self.image, points, 3), resize=True)
        if not self.predictor.is_image_set:
            self.predictor.set_image(np.clip(self.image * 255, 0, 255).astype(np.uint8))
        self.mask_data = self.predictor.process_points(points)
        self.tree2d.insert_batch(self.mask_data)
        # self.tree2d.update_tree()
        # self.tree2d.remove_not_in_tree()
        self.levels_2d = self.tree2d.get_levels()
        self.show_tree_update()
        return True

    def run_tree_seg_2d_post(self):
        super().run_tree_seg_2d_post()
        self._need_update_2d = True
        # fast segment
        # save_path = self.image_dir.joinpath('../my_mask').joinpath(self.image_paths[self.image_index])
        # print(save_path)
        # save_path.parent.mkdir(exist_ok=True)
        # mask = self.tree2d.masks[0].cpu().numpy()
        # utils.save_image(save_path, mask)
        # self.change_image(next=True)
        # self.predictor.set_image((self.image * 255).astype(np.uint8))
        # self.switch_edit_mode()
        # self._need_update_2d = True

    def run_edit_2d(self):
        self._need_update_2d = True
        points = self._points * np.array(self.predictor.original_size)[::-1]
        masks, scores, _ = self.predictor.predict(points, self._labels, return_numpy=False)
        order = torch.argsort(scores, descending=True)
        print('[GUI] run edit 2d', utils.show_shape(masks, scores), scores[order])
        self.mask_data = MaskData(masks=masks[order], iou_preds=scores[order])
        self._edit_mask_idx = 0
        self._need_update_2d = True

    def copy_camera_pose(self, to_2d=True):
        if to_2d:
            Tw2v = ops_3d.look_at(self.view_3d.eye, self.view_3d.at, self.view_3d.up).to(self.device)
            self.new_camera_pose(Tw2v=Tw2v)

            levels = self.tree3d.get_levels(self.aux_data_2d)
            # logger.NOTE('levels:', levels)
            # print('aux_data:', utils.show_shape(self.aux_data))
            indices = torch.cat(levels, dim=0)
            background = torch.logical_not(self.aux_data_2d[0][0])
            masks = torch.stack([background] + [self.aux_data_2d[x.item()][0] for x in indices])
            # print('masks:', utils.show_shape(masks))
            self.tree2d.clear(
                masks=masks.to(self.tree2d.device), scores=torch.zeros(masks.shape[0], device=self.tree2d.device))
            # self.tree_data.update_tree()
            # print(utils.show_shape(self.tree_data.data, self.tree_data.parent))
            # print(indices)
            self.tree2d.node_insert(1, 0)
            indices_map = {}
            for i, x in enumerate(indices, 2):
                x = x.item()
                if x == 0:
                    self.tree2d.node_insert(i, 0)
                else:
                    self.tree2d.node_insert(i, indices_map[self.tree3d.parent[x].item()])
                indices_map[x] = i
            # logger.NOTE('tree_data levels:', self.tree_data.get_levels())
            self.levels_2d = self.tree2d.get_levels()
            dpg.get_item_callback('level1')()
        elif self.Tw2v is not None:
            self.view_3d.set_pose(Tv2w=self.Tw2v.inverse())

    def merge_to_3d(self, *, save=None):
        if self.mask_data is not None and self.tree2d.data is not None:
            self.tree3d.update(self.tree2d.data, self.aux_data_2d)
            if save is None:
                save = dpg.get_value('save_tree_data')
            if save:
                num = len(list(self.cache_dir.glob('*.data'))) + 1
                torch.save(
                    {
                        'tree_data': self.tree2d.save(filename=None),
                        'image': self.image,
                        'tri_id': self.tri_id,
                        'Tw2v': self.Tw2v,
                    }, self.cache_dir.joinpath(f'{num:04d}.data'))
                dpg.set_item_label('load_data', f"Load ({num})")
                print(f'[GUI] save data, index={num}')
            dpg.get_item_callback(f"depth{self.now_level_3d}")()
            self.show_tree_update()

    def run_tree_3d_cycle(self):
        N = dpg.get_value('N_cycle')
        Tw2v = self.tree3d.proposal_camera_pose_cycle(
            N,
            radius_range=(dpg.get_value('radius_min'), dpg.get_value('radius_max')),
            elev_range=(dpg.get_value('theta_min'), dpg.get_value('theta_max')),
            azim_range=(dpg.get_value('phi_min'), dpg.get_value('phi_max')),
        )
        print(f'[GUI] run_tree_3d_cycle: {0}/{N}')
        for i in range(N):
            dpg.set_value('progress', f'run_tree_3d_cycle: {i}/{N}')
            self.new_camera_pose(Tw2v=Tw2v[i])
            if dpg.get_value('dry_run'):
                print(f'[GUI] dry run tree_3d_cycle: {i + 1}/{N}')
                sleep(10 / N)
                continue
            self.autorun_tree_seg_2d()
            self.merge_to_3d()
            print(f'[GUI] run tree_3d_cycle: {i + 1}/{N}')
        dpg.set_value('progress', '')

    def run_tree_3d_uniform(self):
        N = dpg.get_value('N_uniform')
        Tw2v = self.tree3d.proposal_camera_pose_uniform(
            N,
            radius_range=(self.get_value('radius_min'), self.get_value('radius_max')),
            elev_range=(self.get_value('theta_min'), self.get_value('theta_max')),
            azim_range=(self.get_value('phi_min'), self.get_value('phi_max')),
        )
        print(f'[GUI] run_tree_3d_uniform: {0}/{N}')
        for i in range(N):
            dpg.set_value('progress', f'run_tree_3d_uniform: {i}/{N}')
            self.new_camera_pose(Tw2v=Tw2v[i])
            if dpg.get_value('dry_run'):
                print(f'[GUI] dry run tree_3d_uniform: {i + 1}/{N}')
                sleep(1.0)
                continue
            self.autorun_tree_seg_2d()
            self.merge_to_3d()
            print(f'[GUI] run tree_3d_uniform: {i + 1}/{N}')
        dpg.set_value('progress', '')

    def run_tree_3d_grid(self):
        N = dpg.get_value('N_grid')
        Tw2v = self.tree3d.proposal_camera_pose_spherical_grid(N, radius_range=(2.5, 3.0))
        print(Tw2v.shape)
        for i in range(N):
            dpg.set_value('progress', f'run_tree_3d_grid: {i}/{N}')
            self.new_camera_pose(Tw2v=Tw2v[i])
            if dpg.get_value('dry_run'):
                print(f'[GUI] dry run run_tree_3d_grid: {i + 1}/{N}')
                continue
            self.autorun_tree_seg_2d()
            self.merge_to_3d()
            print(f'[GUI] run_tree_3d_grid: {i + 1}/{N}')
        dpg.set_value('progress', '')

    def run_tree_3d_load(self):
        filenames = sorted(list(self.cache_dir.glob('*.data')))
        num_load = dpg.get_value('num_load')
        for i, filename in enumerate(filenames):
            if 0 <= num_load <= i:
                break
            dpg.set_value('progress', f'run_tree_3d_load: {i}/{len(filenames)}')
            data = torch.load(filename, map_location=self.device)
            self.tri_id = data['tri_id']
            self._image = data['image']
            self.Tw2v = data['Tw2v']
            self.reset_2d()
            self.tree2d.load(filename=None, **data['tree_data'])
            self.mask_data = None
            self.levels_2d = self.tree2d.get_levels()
            dpg.get_item_callback('level1')()
            self.merge_to_3d(save=False)
        dpg.set_value('progress', '')

    def run_edit_3d(self):
        self.tree3d.run(epochs=self.get_value('E3D_epochs', 10000), N_view=self.get_value('E3D_num_load', -1))

    def set_edit_3d_threshold(self):
        if self.mode == 'E3D':
            self.tree3d.set_score_threshold(dpg.get_value('edit_threshold'))
            self.view_3d.need_update = True

    def show_uv_results(self):
        pass

    def callback_mouse_hover(self, sender, app_data):
        if self._edit:
            return
        if dpg.is_item_hovered(self.view_3d.image_tag):
            if self.now_level_3d == 0:
                return
            x, y = self.view_3d.get_mouse_pos()
            radiu = 5
            if self._3d_mask is not None:
                H1, W1 = self._3d_mask.shape
                W2, H2 = self.view_3d.size
                x, y = int(x * W1 / W2), int(y * H1 / H2)
                mask_ids = self._3d_mask[max(0, y - radiu):y + radiu, max(0, x - radiu):x + radiu]
                mask_ids, counts = np.unique(mask_ids, return_counts=True)
                mask_id = mask_ids[np.argmax(counts)] - 1

                if mask_id < 0:
                    self.view_3d.data = self.view_3d.origin_data
                else:
                    # may error when change level
                    if 1 <= self.now_level_3d < len(self.levels_3d):
                        nodes = self.levels_3d[self.now_level_3d]
                        if 0 <= mask_id < len(nodes):
                            mask = self.aux_data_3d[nodes[mask_id].item()][0]
                            self.view_3d.data = image_add_mask_boundary(self.view_3d.origin_data, mask)
        else:
            if self.view_3d._origin_data is not None:
                self.view_3d.data = self.view_3d._origin_data
                self.view_3d._origin_data = None
        if dpg.is_item_hovered(self.view_2d.image_tag):
            x, y = self.view_2d.get_mouse_pos()
            radiu = 5
            if self._2d_mask is not None:
                H1, W1 = self._2d_mask.shape
                W2, H2 = self.view_2d.size
                x, y = int(x * W1 / W2), int(y * H1 / H2)
                mask_ids = self._2d_mask[max(0, y - radiu):y + radiu, max(0, x - radiu):x + radiu]
                mask_ids, counts = np.unique(mask_ids, return_counts=True)
                mask_id = mask_ids[np.argmax(counts)] - 1

                if mask_id < 0:
                    self.view_2d.data = self.view_2d.origin_data
                    return
                if self.now_level_2d == 0:
                    mask = self.mask_data['masks'][mask_id]
                else:
                    mask = self.tree2d.masks[self.levels_2d[self.now_level_2d][mask_id] - 1]
                self.view_2d.data = image_add_mask_boundary(self.view_2d.origin_data, mask)
        else:
            if self.view_2d._origin_data is not None:
                self.view_2d.data = self.view_2d._origin_data
                self.view_2d._origin_data = None
        if dpg.is_item_hovered(self.view_seg.image_tag):
            x, y = self.view_seg.get_mouse_pos()
            radiu = 5
            if self._2d_mask is not None:
                H1, W1 = self._2d_mask.shape
                W2, H2 = self.view_seg.size
                x, y = int(x * W1 / W2), int(y * H1 / H2)
                mask_ids = self._2d_mask[max(0, y - radiu):y + radiu, max(0, x - radiu):x + radiu]
                mask_ids, counts = np.unique(mask_ids, return_counts=True)
                # print('view seg', (x, y), mask_ids, counts)
                mask_id = mask_ids[np.argmax(counts)] - 1

                if mask_id < 0:
                    self.view_seg.data = self.view_seg.origin_data
                    return
                if self.now_level_2d == 0:
                    mask = self.mask_data['masks'][mask_id]
                else:
                    mask = self.tree2d.data['masks'][self.levels_2d[self.now_level_2d][mask_id] - 1]
                # mask = mask.cpu().numpy().astype(self.view_seg.data.dtype)
                # mask = cv2.resize(mask, self.view_seg.size, interpolation=cv2.INTER_AREA)
                # print('mask:', mask.shape)
                self.view_seg.data = image_add_mask_boundary(self.view_seg.origin_data, mask)
        else:
            if self.view_seg._origin_data is not None:
                self.view_seg.data = self.view_seg._origin_data
                self.view_seg._origin_data = None
                # print('restore origin data')

    def callback_mouse_wheel(self, sender, app_data):
        self.view_3d.callback_mouse_wheel(sender, app_data)
        if dpg.is_item_hovered(self.view_2d.win_tag):
            x, y = self.view_2d.get_mouse_pos()
            if self._edit and self.mask_data is not None:
                self._edit_mask_idx += app_data * 0.5
                # print('change edit mask index:', self._edit_mask_idx, self.mask_data['masks'].shape)
                self._need_update_2d = True

    def callback_mouse_click(self, sender, app_data):
        if dpg.is_item_left_clicked(self.view_seg.image_tag):
            if self._2d_mask is None:
                return
            x, y = self.view_seg.get_mouse_pos()
            H1, W1 = self._2d_mask.shape
            W2, H2 = self.view_seg.size
            x, y = int(x * W1 / W2), int(y * H1 / H2)
            mask_id = self._2d_mask[y, x].item() - 1
            if self.now_level_2d > 0 and mask_id >= 0:
                mask_id = self.levels_2d[self.now_level_2d][mask_id - 1].item()
            if mask_id > 0:
                print('click:', mask_id)
                self.choose_mask = mask_id + 1
                self._need_update_2d = self.mode != 'E3D'
        if dpg.is_item_left_clicked(self.view_2d.image_tag):
            x, y = self.view_2d.get_mouse_pos()
            if self.mode == 'E2D' and self._edit:
                W, H = self.view_2d.size
                x, y = x / W, y / H
                if self._points is None:
                    self._points = np.array([[x, y]], dtype=np.float32)
                    self._labels = np.array([True], dtype=bool)
                else:
                    self._points = np.append(self._points, [[x, y]], axis=0)
                    self._labels = np.append(self._labels, [True], axis=0)
                self.run_edit_2d()
            elif self._2d_mask is not None:
                H1, W1 = self._2d_mask.shape
                W2, H2 = self.view_2d.size
                x, y = int(x * W1 / W2), int(y * H1 / H2)
                mask_id = self._2d_mask[y, x].item()
                if self.now_level_2d > 0 and mask_id > 0:
                    mask_id = self.levels_2d[self.now_level_2d][mask_id - 1].item()
                if mask_id > 0:
                    print('click:', mask_id)
                    self.choose_mask = mask_id
                    self._need_update_2d = self.mode != 'E3D'
        if dpg.is_item_right_clicked(self.view_2d.image_tag):
            x, y = self.view_2d.get_mouse_pos()
            if self.mode == 'E2D' and self._edit:
                W, H = self.view_2d.size
                x, y = x / W, y / H
                if self._points is None:
                    self._points = np.array([[x, y]], dtype=np.float32)
                    self._labels = np.array([False], dtype=bool)
                else:
                    self._points = np.append(self._points, [[x, y]], axis=0)
                    self._labels = np.append(self._labels, [False], axis=0)
                self.run_edit_2d()

        # if dpg.is_item_left_clicked(self.view_3d.image_tag):
        #     if self._2d_mask is None:
        #         return
        #     x, y = self.view_seg.get_mouse_pos()
        #     H1, W1 = self._2d_mask.shape
        #     W2, H2 = self.view_seg.size
        #     x, y = int(x * W1 / W2), int(y * H1 / H2)
        #     mask_id = self._2d_mask[y, x].item() - 1
        #     if self.now_level_3d > 0 and mask_id >= 0:
        #         mask_id = self.levels_3d[self.now_level_3d][mask_id - 1].item()
        #     if mask_id > 0:
        #         print('click:', mask_id)
        #         self.choose_mask = mask_id + 1
        if self.mode == 'S3D':
            return
        # if not dpg.is_item_left_clicked('control'):
        #     return
        for nodes in (self.levels_2d if self.mode == 'E2D' else self.levels_3d):
            for x in nodes:
                x = x.item()
                if x != 0 and dpg.is_item_left_clicked(f"tree_{x}"):
                    print('click item', x)
                    self.choose_mask = x
            if self.mode == 'E3D':
                self.view_3d.need_update = True
            elif self.mode == 'E2D':
                self._need_update_2d = True

    def _keypress_edit_2d(self):
        if dpg.is_key_pressed(dpg.mvKey_1) or dpg.is_key_pressed(dpg.mvKey_NumPad1):
            self.now_level_2d = 1
        elif dpg.is_key_pressed(dpg.mvKey_2) or dpg.is_key_pressed(dpg.mvKey_NumPad2):
            self.now_level_2d = 2
        elif dpg.is_key_pressed(dpg.mvKey_3) or dpg.is_key_pressed(dpg.mvKey_NumPad3):
            self.now_level_2d = 3
        elif dpg.is_key_pressed(dpg.mvKey_4) or dpg.is_key_pressed(dpg.mvKey_NumPad4):
            self.now_level_2d = 4
        elif dpg.is_key_pressed(dpg.mvKey_5) or dpg.is_key_pressed(dpg.mvKey_NumPad5):
            self.now_level_2d = 5
        elif dpg.is_key_pressed(dpg.mvKey_6) or dpg.is_key_pressed(dpg.mvKey_NumPad6):
            self.now_level_2d = 6
        elif dpg.is_key_pressed(dpg.mvKey_7) or dpg.is_key_pressed(dpg.mvKey_NumPad7):
            self.now_level_2d = 7
        elif dpg.is_key_pressed(dpg.mvKey_8) or dpg.is_key_pressed(dpg.mvKey_NumPad8):
            self.now_level_2d = 8
        elif dpg.is_key_pressed(dpg.mvKey_9) or dpg.is_key_pressed(dpg.mvKey_NumPad9):
            self.now_level_2d = 9
        elif dpg.is_key_pressed(dpg.mvKey_0) or dpg.is_key_pressed(dpg.mvKey_NumPad0):
            self.now_level_2d = 0
        if dpg.is_key_pressed(dpg.mvKey_Back):
            if self._points is not None:
                self._points = self._points[:-1]
                self._labels = self._labels[:-1]
                self.run_edit_2d()
        elif dpg.is_key_pressed(dpg.mvKey_D) or dpg.is_key_pressed(dpg.mvKey_Delete):
            if self.choose_mask > 0 and self.now_level_2d > 0:
                self.tree2d.node_delete(self.choose_mask, move_children=True)
                self.levels_2d = self.tree2d.get_levels()
                self.choose_mask = 0
                self._need_update_2d = True
                print(f'[GUI] Delete node {self.choose_mask} from tree')
        elif dpg.is_key_pressed(dpg.mvKey_A) or dpg.is_key_pressed(dpg.mvKey_Insert):
            if self._edit and self.mask_data is not None:
                idx = int(self._edit_mask_idx) % len(self.mask_data['masks'])
                index = self.tree2d.insert(mask=self.mask_data['masks'][idx], score=self.mask_data['iou_preds'][idx])
                self.levels_2d = self.tree2d.get_levels()
                self.now_level_2d = self.tree2d.get_depth(index)
                self.choose_mask = 0 if index <= 0 else index
                self.switch_edit_mode()
                print(f'[GUI] Add new node {index} to tree')
        elif dpg.is_key_pressed(dpg.mvKey_C):
            if self._edit:
                self._points = None
                self._labels = None
                self.mask_data = None
                self._need_update_2d = True
                print(f'[GUI] Clear all points')
        elif dpg.is_key_pressed(dpg.mvKey_S):
            if self.choose_mask > 0 and self._edit and self.mask_data is not None:
                new_mask = self.mask_data['masks'][int(self._edit_mask_idx) % len(self.mask_data['masks'])]
                old_mask = self.tree2d.masks[self.choose_mask - 1]
                new_mask = old_mask & (~new_mask)
                self.tree2d.masks[self.choose_mask - 1] = new_mask
                self.tree2d.areas[self.choose_mask - 1] = new_mask.sum()
                self.tree2d.node_delete(self.choose_mask, move_children=True)
                index = self.tree2d.insert(new_mask, i=self.choose_mask)
                self.levels_2d = self.tree2d.get_levels()
                self.choose_mask = max(0, index)
                self.switch_edit_mode()
                print(f'[GUI] Subtract mask from node {self.choose_mask} in tree')
        elif dpg.is_key_pressed(dpg.mvKey_F):
            if self.choose_mask > 0 and self._edit and self.mask_data is not None:
                new_mask = self.mask_data['masks'][int(self._edit_mask_idx) % len(self.mask_data['masks'])]
                old_mask = self.tree2d.masks[self.choose_mask - 1]
                new_mask = new_mask | old_mask
                self.tree2d.masks[self.choose_mask - 1] = new_mask
                self.tree2d.areas[self.choose_mask - 1] = new_mask.sum()
                self.tree2d.node_delete(self.choose_mask, move_children=True)
                index = self.tree2d.insert(new_mask, i=self.choose_mask)
                self.levels_2d = self.tree2d.get_levels()
                self.choose_mask = max(0, index)
                self.switch_edit_mode()
                print(f'[GUI] Fusion mask from node {self.choose_mask} in tree')
        elif dpg.is_key_pressed(dpg.mvKey_R):
            self.tree2d.remove_not_in_tree()
            self.tree2d.reset()
            self.tree2d.update_tree()
            self.tree2d.remove_not_in_tree()
            self.levels_2d = self.tree2d.get_levels()
            print('[GUI] Rebuild Tree2D')
        elif dpg.is_key_pressed(dpg.mvKey_Spacebar):
            self.run_tree_seg_2d_post()
        return

    def callback_keypress(self, sender, app_data):
        # print('sender:', sender, 'app_data:', app_data)
        if self.mode == 'E3D':
            cnt = self.tree3d.cnt
        elif self.mode == 'E2D':
            cnt = self.tree2d.cnt
        else:
            cnt = 0
        if dpg.is_key_pressed(dpg.mvKey_Left):
            if self.mode != 'S3D' and cnt > 0:
                choose_mask = 1 if self.choose_mask <= 0 else self.choose_mask
                init_index = choose_mask
                while True:
                    choose_mask -= 1
                    if choose_mask <= 0:
                        choose_mask = cnt
                    if dpg.does_alias_exist(f"tree_{choose_mask}"):
                        break
                    if choose_mask == init_index:
                        choose_mask = 0
                        break
                self.choose_mask = choose_mask
            else:
                self.choose_mask = 0
            if self.mode == 'E3D':
                self.view_3d.need_update = True
            elif self.mode == 'E2D':
                self._need_update_2d = True
        elif dpg.is_key_pressed(dpg.mvKey_Right):
            if self.mode != 'S3D' and cnt > 0:
                choose_mask = cnt if self.choose_mask <= 0 else self.choose_mask
                init_index = choose_mask
                while True:
                    choose_mask += 1
                    if choose_mask > cnt:
                        choose_mask = 1
                    if dpg.does_alias_exist(f"tree_{choose_mask}"):
                        break
                    if choose_mask == init_index:
                        choose_mask = 0
                        break
                self.choose_mask = choose_mask
            else:
                self.choose_mask = 0
            if self.mode == 'E3D':
                self.view_3d.need_update = True
            elif self.mode == 'E2D':
                self._need_update_2d = True
        if self.mode == 'E2D':
            self._keypress_edit_2d()
        if dpg.is_key_pressed(dpg.mvKey_E):
            self.switch_edit_mode()
            self._need_update_2d = True
        if dpg.is_key_pressed(dpg.mvKey_H):
            if dpg.get_item_state('win_help')['visible']:
                dpg.hide_item('win_help')
            else:
                dpg.show_item('win_help')

    def callback_exit(self):
        self.save_ini()
        dpg.minimize_viewport()
        console.print(f"[red]Debug callback exit")

    def switch_edit_mode(self):
        if self._edit:
            self._edit = False
            dpg.bind_item_theme('edit_mode', self.default_theme)
        else:
            self._edit = True
            dpg.bind_item_theme('edit_mode', self.choose_theme)
        self.mask_data = None
        self._points = None
        self._labels = None
        self._need_update_2d = True

    def make_control_panel(self):
        with dpg.group(horizontal=True):
            dpg.add_button(label='Edit 2D', tag='E2D', width=60, height=30, callback=self._set_value('mode', 'E2D'))
            dpg.add_button(label='Edit 3D', tag='E3D', width=60, height=30, callback=self._set_value('mode', 'E3D'))
            dpg.add_button(label='Seg 3D', tag='S3D', width=60, height=30, callback=self._set_value('mode', 'S3D'))
            dpg.bind_item_theme('E2D', self.default_theme)
            dpg.bind_item_theme('E3D', self.default_theme)
            dpg.bind_item_theme('S3D', self.default_theme)

            # dpg.add_text('Model')
            dpg.add_button(label=self._model_type, tag='change_model', height=30)

            def change_model(name='SAM'):

                def change(*args):
                    if self._model_type != name:
                        self._predictor = None
                        self._model = None
                    dpg.configure_item('win_change_model', show=False)
                    self._model_type = name
                    dpg.configure_item('change_model', label=self._model_type)
                    print(f'[GUI] change model to {self._model_type}')
                    self.save_ini()
                    return

                return change

            with dpg.popup(dpg.last_item(), mousebutton=dpg.mvMouseButton_Left, modal=False, tag='win_change_model'):
                dpg.add_button(label='SAM', callback=change_model('SAM'))
                dpg.add_button(label='SAM-L', callback=change_model('SAM-L'))
                dpg.add_button(label='SAM-B', callback=change_model('SAM-B'))
                dpg.add_button(label='Semantic-SAM-L', callback=change_model('Semantic-SAM-L'))
                dpg.add_button(label='Semantic-SAM-T', callback=change_model('Semantic-SAM-T'))

            self.make_bottom_status_options()
        dpg.add_separator()
        with dpg.collapsing_header(label="3D Viewer", default_open=False, tag='ctl_view_3d'):
            with dpg.group(horizontal=True):
                dpg.add_text('fovy')
                dpg.add_slider_float(
                    min_value=15.,
                    max_value=180.,
                    default_value=math.degrees(self.view_3d.fovy),
                    callback=lambda *args: self.view_3d.set_fovy(dpg.get_value('set_fovy')),
                    tag='set_fovy')

            def change_eye(*args):
                print('change camera position', args)
                self.view_3d.eye = self.view_3d.eye.new_tensor(
                    [dpg.get_value(item) for item in ['eye_x', 'eye_y', 'eye_z']])
                self.view_3d.need_update = True

            with dpg.group(horizontal=True):
                dpg.add_text('camera pos:')
                dpg.add_button(label='change', callback=change_eye)

            with dpg.group(horizontal=True):
                dpg.add_text('x')
                dpg.add_input_float(tag='eye_x', width=100)
                dpg.add_text('y')
                dpg.add_input_float(tag='eye_y', width=100)
                dpg.add_text('z')
                dpg.add_input_float(tag='eye_z', width=100)

        with dpg.collapsing_header(label='Predictor Options', default_open=False, tag='ctl_predictor'):
            self.make_predictor_options()
        with dpg.collapsing_header(label='Edit 2D Options', default_open=True, tag='ctl_edit_2d'):
            self.make_edit_2d_options()
        with dpg.collapsing_header(label='2D Tree Segmentation Options', default_open=True, tag='ctl_tree_2d'):
            self.make_2d_tree_segmentatin_options()
        with dpg.collapsing_header(label='3D Tree Segmentation Options', default_open=True, tag='ctl_tree_3d'):
            self.make_3d_tree_segmentation_options()
        with dpg.collapsing_header(label='3D Edit Options', default_open=True, tag='ctl_edit_3d', show=False):
            self.make_3d_edit_options()

        def fit_callback():
            if self.Tw2v is None:
                return
            Tv2w = self.Tw2v.inverse()
            scale = fit_window(self.tri_id)
            print('[GUI] scale=', scale)
            Tv2w[:3, 3] *= scale
            self.new_camera_pose(Tw2v=Tv2w.inverse())

        # dpg.add_button(label='Fit', callback=fit_callback)

        dpg.add_collapsing_header(label='root', tag='tree_0', show=False)

    def make_predictor_options(self):
        with dpg.group(horizontal=True):
            dpg.add_text('Threshold: iou=')
            o21 = dpg.add_input_float(tag='pred_iou_thresh', default_value=0.88, width=80, step=0)
            dpg.add_text('stability=')
            o22 = dpg.add_input_float(tag='stability_score_thresh', default_value=0.95, width=80, step=0)
            dpg.add_text('nms=')
            o23 = dpg.add_input_float(tag='box_nms_thresh', default_value=0.7, width=80, step=0)

        with dpg.group(horizontal=True):
            t1 = dpg.add_text('Merge: in=')
            o31 = dpg.add_input_float(default_value=0.8, step=0, tag='in_threshold')
            t2 = dpg.add_text('in area=')
            o32 = dpg.add_input_int(default_value=10, step=0, tag='in_area_threshold')
            t3 = dpg.add_text('union=')
            o33 = dpg.add_input_float(default_value=0.1, step=0, tag='union_threshold')
            t4 = dpg.add_text('min area=')
            o34 = dpg.add_input_int(default_value=100, step=0, tag='min_area')
            # print(dpg.get_item_width('control') , list(dpg.get_item_width(t) for t in [t1, t2, t3, t4]))
            # width = (dpg.get_item_width('control') - sum(dpg.get_item_width(t) for t in [t1, t2, t3, t4])) // 4
            width = 50
            for e in [o31, o32, o33, o34]:
                dpg.set_item_width(e, width)

        with dpg.group(horizontal=True):
            dpg.add_text('Sample: max steps=')
            dpg.add_input_int(tag='max_steps', default_value=100, width=100)
            dpg.add_text('filled')
            dpg.add_input_float(tag='filled_threshold', default_value=0.9, format='%.3f', width=100, step=0.005)

        with dpg.group(horizontal=True):
            dpg.add_text('num points: stage1:')
            o11 = dpg.add_input_int(default_value=32, width=100, tag='points_per_side')
            dpg.add_text('stage2:')
            o12 = dpg.add_input_int(default_value=256, width=100, tag='points_per_update')

        with dpg.group(horizontal=True):
            dpg.add_text('alpha')
            dpg.add_slider_float(min_value=0, max_value=1., default_value=0.5, tag='alpha', callback=self.update_viewer)

        def changed_callback(item):

            def changed(*args):
                if self._predictor is not None:
                    setattr(self._predictor, item, dpg.get_value(item))
                    print(f'[Tree 2D] Change {item} to {getattr(self._predictor, item)}')

            return changed

        for item in [o11, o12, o21, o22, o23]:
            print(item)
            dpg.set_item_callback(item, changed_callback(item))

    def make_edit_2d_options(self):
        with dpg.file_dialog(
                directory_selector=False,
                show=False,
                callback=self.change_image,
                id="choose_image",
                width=700,
                height=400,
                default_path=self.image_dir.as_posix(),
        ):
            dpg.add_file_extension("Image{.png,.jpg,.jpeg,.tif}", custom_text='[Image]')
        with dpg.group(horizontal=True):
            # with dpg.table():
            dpg.add_button(label='change', callback=lambda: dpg.show_item("choose_image"))
            dpg.add_button(label='last', callback=lambda: self.change_image(last=True))
            dpg.add_text(label='no image', tag='image_path')
            dpg.add_button(label='next', callback=lambda: self.change_image(next=True))

        with dpg.group(horizontal=True):
            dpg.add_button(label='save', callback=self.save_result)
            dpg.add_button(label='load', callback=self.load_result)
            dpg.add_text('size min=')
            dpg.add_input_int(
                default_value=256,
                step=0,
                tag='min_image_size',
                width=60,
                min_clamped=True,
                min_value=128,
                callback=self._cbw(self.change_image))
            dpg.add_text('max=')
            dpg.add_input_int(
                default_value=1024,
                step=0,
                tag='max_image_size',
                width=60,
                min_clamped=True,
                min_value=128,
                callback=self._cbw(self.change_image))
            dpg.add_text('show points:')
            dpg.add_checkbox(tag='show_points', callback=self._set_value('_need_update_2d', True), default_value=True)

        with dpg.group(horizontal=True):
            W, H = 70, 40
            dpg.add_button(width=W, height=H, label='Reset', callback=self.reset_2d)
            # dpg.add_button(width=W, height=H, label='New Pose', callback=self.new_camera_pose)
            dpg.add_button(width=W, height=H, label='Stage1', callback=self.run_tree_seg_2d_stage1)
            dpg.add_button(width=W, height=H, label='Stage2', callback=self.run_tree_seg_2d_stage2)
            dpg.add_button(width=W, height=H, label='Post', callback=self.run_tree_seg_2d_post)
            dpg.add_button(width=W, height=H, label='AutoRun', callback=self.autorun_tree_seg_2d)
            dpg.add_button(width=W, height=H, label='Edit', tag='edit_mode', callback=self.switch_edit_mode)

    def make_2d_tree_segmentatin_options(self):
        with dpg.group(horizontal=True):
            W, H = 70, 40
            dpg.add_button(width=W, height=H, label='Reset', callback=self.reset_2d)
            dpg.add_button(width=W, height=H, label='New Pose', callback=self.new_camera_pose)
            dpg.add_button(width=W, height=H, label='Stage1', callback=self.run_tree_seg_2d_stage1)
            dpg.add_button(width=W, height=H, label='Stage2', callback=self.run_tree_seg_2d_stage2)
            dpg.add_button(width=W, height=H, label='Post', callback=None)
            dpg.add_button(width=W, height=H, label='AutoRun', callback=self.autorun_tree_seg_2d)

    def make_3d_tree_segmentation_options(self):

        def update_view_3d(*args):
            self.view_3d.need_update = True
            # print('need update view3d')

        with dpg.file_dialog(
                # directory_selector=False,
                show=False,
                callback=lambda sender, app_data: self.load_mesh(app_data['file_path_name']),
                id="choose_mesh",
                width=700,
                height=400,
                default_path='/home/wan/data/meshes',
                modal=True,
        ):
            dpg.add_file_extension("Mesh{.obj,.ply}", custom_text='mesh')
            with dpg.group(horizontal=True):
                dpg.add_checkbox(tag='mesh_load_all')
                dpg.add_text('Load all parts')
                dpg.add_checkbox(tag='mesh_load_force')
                dpg.add_text('force')
            with dpg.child_window(tag='win_history'):
                dpg.add_text('History')

                def _choose_history(sender, app_data):
                    # print(sender, app_data, dpg.get_value(app_data[1]))
                    # dpg.set_value('history_detail', app_data)
                    # print(dpg.get_item_configuration('choose_mesh'))
                    self.load_mesh(dpg.get_value(app_data[1]))
                    dpg.hide_item('choose_mesh')
                    # dpg.configure_item('choose_mesh', default_filename=dpg.get_value(app_data[1]))

                with dpg.item_handler_registry(tag='history_handler'):
                    dpg.add_item_clicked_handler(button=dpg.mvMouseButton_Left, callback=_choose_history)

                for i in range(10):
                    dpg.add_text(tag=f'history_{i}', wrap=210)
                    dpg.bind_item_handler_registry(f'history_{i}', 'history_handler')

        with dpg.group(horizontal=True):
            dpg.add_button(label='change mesh', callback=lambda: dpg.show_item("choose_mesh"))
            dpg.add_button(label='Load', callback=self.load_result)
            dpg.add_button(label='Save', callback=self.save_result)

        W, H = 90, 40
        with dpg.table(header_row=False, resizable=False, width=512):
            dpg.add_table_column()
            dpg.add_table_column()
            dpg.add_table_column()
            dpg.add_table_column()
            dpg.add_table_column()
            with dpg.table_row():
                with dpg.table_cell():
                    dpg.add_button(label='Reset', callback=self.reset_3d, height=H, width=W)
                with dpg.table_cell():
                    dpg.add_button(label='Merge', callback=self.merge_to_3d, height=H, width=W)
                with dpg.table_cell():
                    dpg.add_button(label='Cycle', callback=self.run_tree_3d_cycle, height=H, width=W)
                with dpg.table_cell():
                    dpg.add_button(label='Uniform', callback=self.run_tree_3d_uniform, height=H, width=W)
                with dpg.table_cell():
                    dpg.add_button(label='Grid', callback=self.run_tree_3d_grid, height=H, width=W)
                # with dpg.table_cell():
                #     def compress():
                #         if self._tree_3d is not None:
                #             self.tree3d.compress()
                #
                #     dpg.add_button(label='Compress', callback=compress, height=H, width=W)

            with dpg.table_row():
                with dpg.table_cell():
                    dpg.add_text('raduis range:')
                with dpg.table_cell():
                    dpg.add_text('Num Cameras:')
                with dpg.table_cell():
                    dpg.add_input_int(tag='N_cycle', default_value=10, width=W)
                with dpg.table_cell():
                    dpg.add_input_int(tag='N_uniform', default_value=3, width=W)
                with dpg.table_cell():
                    dpg.add_input_int(tag='N_grid', default_value=3, width=W)

            with dpg.table_row():
                kwargs = dict(step=0, width=W)
                with dpg.table_cell():
                    dpg.add_input_float(tag='radius_min', default_value=2.5, **kwargs, min_clamped=True, min_value=0.5)
                    dpg.add_input_float(tag='radius_max', default_value=2.5, **kwargs, min_clamped=True, min_value=0.5)
                with dpg.table_cell():
                    dpg.add_text('theta min:')
                    dpg.add_text('theta max:')
                with dpg.table_cell():
                    dpg.add_input_float(default_value=0., tag='theta_min', **kwargs)
                    dpg.add_input_float(default_value=180., tag='theta_max', **kwargs)
                with dpg.table_cell():
                    dpg.add_text('phi min:')
                    dpg.add_text('phi max:')
                with dpg.table_cell():
                    dpg.add_input_float(default_value=0., tag='phi_min', **kwargs)
                    dpg.add_input_float(default_value=360., tag='phi_max', **kwargs)

            with dpg.table_row():
                # with dpg.table_cell():
                #     dpg.add_button(label='Reload', callback=self.reload, height=H, width=W)
                with dpg.table_cell():
                    dpg.add_checkbox(tag='save_tree_data', label='save')
                    dpg.add_checkbox(tag='dry_run', label='dry run', default_value=False)
                with dpg.table_cell():
                    num = len(list(self.cache_dir.glob('*.data')))
                    dpg.add_button(
                        label=f'Load ({num})', tag='load_data', width=W, height=H, callback=self.run_tree_3d_load)

                def clean_all_data(*args):
                    utils.dir_create_and_clear(self.cache_dir, '*.data')
                    dpg.set_item_label('load_data', "Load (0)")

                with dpg.table_cell():
                    dpg.add_button(label='clean', width=W, height=H, callback=clean_all_data)
        # with dpg.group(horizontal=True):
        #     dpg.add_text('convert camera pose')
        #     dpg.add_button(label='3D to 2D', callback=lambda: self.copy_camera_pose(True))
        #     dpg.add_button(label='2D to 3D', callback=lambda: self.copy_camera_pose(False))

        dpg.add_input_int(label='load num', default_value=-1, tag='num_load')

        dpg.add_button(
            parent=self.view_3d.win_tag,
            label='>',
            tag='copy_to_2d',
            pos=(512 - 30 - 10, 256),
            width=30,
            height=30,
            callback=lambda: self.copy_camera_pose(to_2d=True))
        dpg.add_button(
            parent=self.view_2d.win_tag,
            label='<',
            tag='copy_to_3d',
            pos=(10, 256),
            width=30,
            height=30,
            callback=lambda: self.copy_camera_pose(to_2d=False))
        dpg.add_button(
            parent=self.view_3d.win_tag,
            label='UV',
            tag='get_uv',
            pos=(512 - 30 - 10, 210),
            width=30,
            height=30,
            callback=lambda: self.show_uv_results)

    def make_3d_edit_options(self):
        W, H, P = 50, 40, 10
        with dpg.group(horizontal=True):
            dpg.add_button(label='Mesh', width=W, height=H, callback=lambda: dpg.show_item("choose_mesh"))
            dpg.add_button(label='Save', width=W, height=H, callback=self.save_result)
            dpg.add_button(label='Load', width=W, height=H, callback=self.load_result)
            dpg.add_button(label='Reset', width=W, height=H, callback=self.reset_3d)
            dpg.add_button(label='Run', width=W, height=H, callback=self.run_edit_3d)
        with dpg.group(horizontal=True):
            dpg.add_text('Epochs')
            dpg.add_input_int(default_value=10000, step=0, tag='E3D_epochs', width=80)
            dpg.add_text('Num Load:')
            dpg.add_input_int(default_value=-1, step=0, tag='E3D_num_load', width=80)
        with dpg.group(horizontal=True):
            dpg.add_text('edit_threshold')
            dpg.add_slider_float(
                min_value=0, max_value=1., default_value=0.5, tag='edit_threshold', callback=self.set_edit_3d_threshold)
            dpg.add_button(label='set', callback=self.set_edit_3d_threshold)

    def make_bottom_status_options(self):
        with dpg.group(horizontal=True, tag='status_bar'):
            dpg.add_text(tag='fps')
            dpg.add_text(tag='progress')
            dpg.add_button(label='Exit', callback=exit)
            dpg.add_button(label='Reload', callback=self.reload)

    def make_show_tree_2d_mask_level(self):
        size, pad = 30, 10
        dpg.push_container_stack(self.view_seg.win_tag)
        for i, level in enumerate(range(-1, 10)):
            dpg.add_button(
                label={
                    0: 'E',
                    -1: 'I'
                }[level] if level <= 0 else f"{level}",
                pos=(pad + (size + pad) * i, pad),
                width=size,
                height=size,
                tag=f'level{level}',
                show=False,
                callback=self._set_value('now_level_2d', level),
            )
            dpg.bind_item_theme(f"level{level}", self.default_theme)
        dpg.pop_container_stack()
        dpg.show_item('level-1')
        dpg.show_item('level0')

    def make_show_tree_3d_mask_level(self):
        size = 30
        pad = 10

        def show_level_callback(level):

            def show_level():
                self.now_level_3d = level
                return

            return show_level

        dpg.push_container_stack(self.view_3d.win_tag)
        for level in range(0, 10):
            dpg.add_button(
                label=f'{level}',
                pos=(pad + (size + pad) * level, pad),
                width=size,
                height=size,
                tag=f'depth{level}',
                show=level == 0,
                callback=show_level_callback(level))
            dpg.bind_item_theme(f"depth{level}", self.default_theme)
        dpg.pop_container_stack()

    def add_help_popup(self):
        with dpg.window(label='Help', modal=True, show=False, tag='win_help'):
            dpg.add_text(f'"H": show/hide this help page')
            dpg.add_text(f'"<Left>": choose last node in the tree')
            dpg.add_text(f'"<Right>": choose next node in the tree')
            dpg.add_text(f'"E": Enter/Exit edit mode')
            dpg.add_separator()
            dpg.add_text('Edit Mode')
            dpg.add_text("Left/Right Click: add positive or negative point")
            dpg.add_text(f'"D" or "<Delete>": Delete choosed node in tree')
            dpg.add_text(f'"A" or "<Insert>": Insert new mask into tree')
            dpg.add_text(f'"F": Fuse(Merge) new mask and choosed mask')
            dpg.add_text(f'"S": Subtract new mask from choosed mask')
            dpg.add_text(f'"C" or <ESC>: Clear all points')
            dpg.add_text(f'"R": Rebuild Tree')
            dpg.add_text(f'"<Back>": delete last add point')
            dpg.add_text(f'Mouse Wheel: choose different masks')

    def change_image(self, sender=None, app_data=None, index=None, next=False, last=False):
        print(f"[GUI]", sender, app_data, index, next, last)
        if app_data is not None:
            self.image_dir = Path(app_data['current_path'])
            self.image_paths = []
            for path in self.image_dir.glob('*.*'):
                if path.suffix.lower() in utils.image_extensions:
                    self.image_paths.append(path.name)
            index = self.image_paths.index(app_data['file_name'])
            print(f'[GUI] change work dir to: {self.image_dir}, there are {len(self.image_paths)} images')
        n = len(self.image_paths)
        if n == 0:
            print(f'There are no image in diectory "{self.image_dir}"')
            dpg.set_value('image_path', 'no image')
            dpg.show_item('choose_image')
            dpg.focus_item('choose_image')
            return
        if index is not None:
            self.image_index = index
        elif next:
            self.image_index += 1
            if self.image_index >= n:
                self.image_index -= n
        elif last:
            self.image_index -= 1
            if self.image_index < 0:
                self.image_index = n - 1
        self.reset_2d()
        if self._predictor is not None:
            self.predictor.reset_image()
        image = utils.load_image(self.image_dir.joinpath(self.image_paths[self.image_index]))
        text = self.image_paths[self.image_index]
        if len(text) >= 36:
            text = '...' + text[-33:]
        dpg.set_value('image_path', f"{text:<36} ({self.image_index}/{n})")
        H, W, _ = image.shape
        max_size = self.get_value('max_image_size', 1024)
        min_size = self.get_value('min_image_size', 256)

        scale = min(min(max_size / H, max_size / W), max(max(min_size / H, min_size / W), 1.0))
        H2, W2 = int(H * scale), int(W * scale)
        # image = cv2.resize(image, (W2, H2), interpolation=cv2.INTER_CUBIC)
        print(f'[GUI] load image from: {self.image_dir / self.image_paths[self.image_index]}, shape: {image.shape}')
        self.view_2d.resize(W2, H2, image.shape[-1])
        self.image = image.astype(np.float32) / 255.
        dpg.set_item_pos('control', [W2, 0])
        # dpg.set_item_pos(self.view_2d.image_tag, [0, 30])
        H = max(512, H2 + 30)
        dpg.set_item_height('control', H)
        # dpg.set_item_pos('win_status', [0, H - 30])
        dpg.set_viewport_width(W2 + 512)
        dpg.set_viewport_height(H)
        for i, level in enumerate(range(-1, 10)):
            dpg.set_item_pos(f"level{level}", pos=[10 + (30 + 10) * i, H2])
        self.save_ini()
        self._need_update_2d = True

    def set_default_font(self, name='wqy-zenhei', fontsize=16):
        font_serach_dirs = [
            Path('segment_anything').resolve(),
            Path('/usr/share/fonts'),
            Path('~/.local/share/fonts').expanduser()
        ]
        with dpg.font_registry() as self.font_registry_id:
            for search_path in font_serach_dirs:  # type: Path
                paths = list(search_path.rglob(f'{name}.*'))
                if len(paths) == 0:
                    continue
                with dpg.font(paths[0].as_posix(), fontsize) as default_font:
                    dpg.add_font_range_hint(dpg.mvFontRangeHint_Chinese_Full)
                dpg.bind_font(default_font)
                return
        print('[GUI] Not setting defaulut font')

    def save_result(self):
        prefix = '' if self.mode != 'E2D' else os.path.splitext(self.image_paths[self.image_index])[0]
        suffix = {'E2D': '.tree2d', 'E3D': '.tree3dv2', 'S3D': '.tree3d'}[self.mode]
        save_dir = self.image_dir if self.mode == 'E2D' else self.cache_dir
        saved_names = sorted([path.stem for path in save_dir.glob(f"{prefix}*{suffix}")])
        if not dpg.does_alias_exist('save_popup'):
            dpg.add_window(label="Save Results", modal=True, show=False, tag="save_popup", popup=True, autosize=True)
        dpg.delete_item('save_popup', children_only=True)
        dpg.set_item_pos('save_popup', dpg.get_item_pos('control'))

        def _save():
            filename = dpg.get_value('save_file_name')
            if self.mode == 'E2D' and not filename.startswith(prefix):
                filename = prefix + filename
            elif len(filename) == 0:
                filename = 'gui'
            save_path = save_dir.joinpath(f"{filename}{suffix}")
            if self.mode == 'E2D' and self._tree_2d is not None:
                self._tree_2d.save(save_path)
            else:
                self.tree3d.save(save_path)
            print(f'[GUI]: save tree segment results to {save_path}')
            dpg.hide_item('save_popup')

        for name in saved_names:
            with dpg.group(horizontal=True, parent='save_popup'):
                dpg.add_button(label=name, callback=lambda *, _name=name: dpg.set_value('save_file_name', _name))
        dpg.add_separator(parent='save_popup')
        default_name = prefix
        cnt = 1
        while default_name in saved_names:
            default_name = f"{prefix}_{cnt}"
            cnt += 1
        with dpg.group(horizontal=True, parent='save_popup'):
            dpg.add_input_text(default_value=default_name, tag='save_file_name')
            dpg.add_button(label='save', callback=_save)
        dpg.show_item('save_popup')

    def load_result(self):
        if self.mode == 'E3D':
            load_files = sorted(list(self.cache_dir.glob("*.tree3dv2")))
        elif self.mode == 'S3D':
            load_files = sorted(list(self.cache_dir.glob("*.tree3d")))
        elif self.mode == 'E2D':
            name = os.path.splitext(self.image_paths[self.image_index])[0]
            load_files = sorted(list(self.image_dir.glob(f"{name}*.tree2d")))
        else:
            raise RuntimeError
        # print(f"There are {len(load_files)} results can load")
        # if len(load_files) == 0:
        #     return
        if not dpg.does_alias_exist('load_popup'):
            dpg.add_window(label="Load Results", modal=True, show=False, tag="load_popup", popup=True)

        dpg.delete_item('load_popup', children_only=True)
        dpg.set_item_pos('load_popup', dpg.get_item_pos('control'))

        def _load(load_path: Path):

            def __load():
                if self.mode == 'E2D':
                    self.reset_2d()
                    self.tree2d.load(load_path)
                    self.levels_2d = self.tree2d.get_levels()
                    self.now_level_2d = 1
                elif self.mode == 'E3D':
                    self.tree3d.load(load_path)
                    self.view_3d.need_update = True
                elif self.mode == 'S3D':
                    self.tree3d.load(load_path)
                    self.view_3d.need_update = True
                print(f'[GUI]: load results from {load_path}')
                dpg.hide_item("load_popup")
                self.show_tree_update()

            return __load

        for filepath in load_files:
            with dpg.group(horizontal=True, parent='load_popup'):
                dpg.add_button(label=filepath.name, callback=_load(filepath))
                # dpg.add_button(label='D', callback=lambda: dpg.hide_item("load_popup"))
        dpg.show_item('load_popup')

    def save_ini(self):
        ini_path = self.cache_root.joinpath('tree_3d_seg.ini')
        with ini_path.open('w', encoding='utf-8') as f:
            yaml.dump(
                {
                    'mesh_path': self.mesh_path.as_posix(),
                    '_mode': self._mode,
                    '_model_type': self._model_type,
                    'image_dir': self.image_dir.as_posix(),
                    'image_index': self.image_index,
                    'viewport_pos': dpg.get_viewport_pos(),
                    'mesh_history': self._mesh_history,
                }, f)

    def load_ini(self):
        ini_path = self.cache_root.joinpath('tree_3d_seg.ini')
        if not ini_path.exists():
            return
        with ini_path.open('r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        try:
            self.mesh_path = Path(data['mesh_path'])
            self._mode = data['_mode']
            self._model_type = data['_model_type']
            self.image_dir = Path(data['image_dir'])
            self.image_index = data.get('image_index', 0)
            self.image_paths = [p.name for p in self.image_dir.glob('*') if p.suffix.lower() in utils.image_extensions]
            print('viewport_pos:', data['viewport_pos'])
            dpg.set_viewport_pos(data.get('viewport_pos', [0, 0]))
            self._mesh_history = data.get('mesh_history', [])
        except KeyError as e:
            os.remove(ini_path)
            print(f'[GUI] delete ini file due to unexcepted key', e)
        print('[GUI] load ini file from:', ini_path)

    def update_viewer(self):
        dpg.get_item_callback(f"level{self.now_level_2d}")()


def exit():
    # dpg.stop_dearpygui()
    dpg.destroy_context()


if __name__ == '__main__':
    torch.cuda.set_per_process_memory_fraction(0.7, 0)
    TreeSegmentGUI().run()

# TODO: 标记3D模型转动状态, 并降低转动时的渲染分辨率, 以提高流畅性
# TODO: 窗口大小可只有调整, 底部状态栏
# TODO: sam路径
