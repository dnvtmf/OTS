import os

import matplotlib.pyplot as plt
import yaml
import importlib
import math
from pathlib import Path
from time import time, sleep
from typing import Union, Tuple
import gc
from rich.console import Console

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
import torch
from torch import Tensor

from tree_segmentation.extension import utils, ops_3d, Masks
from tree_segmentation.extension.utils import ImageViewer, Viewer3D
from tree_segmentation import Tree2D
from tree_segmentation.util import get_colored_masks, image_add_mask_boundary


class LabelGUI:
    def __init__(self):
        self.image_dir = Path('~/data/NeRF/D_NeRF/lego/train').expanduser()
        self.seg_2d_dir = Path('~/data/NeRF/D_NeRF/lego/tree_seg/train').expanduser()
        self.cache_dir = Path('./results/D_NeRF/lego')
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.image_paths = sorted(list(self.image_dir.glob('*.png')))
        self.images = np.stack([utils.load_image(path) for path in self.image_paths])
        self.images = self.images.astype(np.float32) / 255.
        self.image_size = (self.images.shape[2], self.images.shape[1])
        self.num_img = len(self.images)

        print(f"Load images {self.images.shape} from {self.image_dir}")
        self.tree2d = []
        for image_path in self.image_paths:
            tree_2d_path = self.seg_2d_dir.joinpath(image_path.stem + '.tree2d')
            tree2d = Tree2D(format=Masks.BINARY)
            if True and tree_2d_path.exists():
                tree2d.load(tree_2d_path)
            else:
                tree2d.load_from_png(self.seg_2d_dir, image_path.stem)
                tree2d.save(tree_2d_path)
            self.tree2d.append(tree2d)
        print(f"Load Tree2d from {self.seg_2d_dir}")
        self.show_mask_ids = [[0] for i in range(self.num_img)]
        self.show_masks = [torch.zeros(self.image_size[::-1], dtype=torch.int) for _ in range(self.num_img)]
        self.choices = np.zeros(self.num_img, dtype=np.int32)  # type: np.ndarray

        # tree
        self._tree_node_now = 0
        self.tree = {0: {
            'parent': 0,
            'masks': np.zeros(self.num_img, dtype=np.int32),
            'depth': 0,
            'children': [],
        }}
        self.tree_node_num = 1

        dpg.create_context()
        dpg.create_viewport(title='Tree Segmentation Labeling', min_width=400, resizable=True)
        self.update_viewer = []

        self.image_index = 0
        self.show_size = [256, 256 * self.image_size[1] // self.image_size[0]]
        self.padding = 10
        self.num_row = 1
        self.num_col = 1
        self.image_crop = [[0., 0., self.show_size[0] / self.image_size[0]] for _ in range(self.num_img)]

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
            self.viewer = ImageViewer(size=(512, 512),
                channels=self.images.shape[-1],
                tag='viewer',
                pos=(0, 0),
                no_resize=True,
                no_move=True)
            with dpg.window(
                tag='control',
                width=256,
                height=512,
                pos=(512, 0),
                no_resize=True,
                no_move=True,
                no_title_bar=True,
            ):
                self.build_control_ui()
        dpg.set_primary_window('Primary Window', True)
        dpg.set_viewport_resize_callback(self.change_windows_size)

        self._last_mouse_pos = None
        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(callback=self.callback_mouse_drag)
            dpg.add_mouse_wheel_handler(callback=self.callback_mouse_wheel)
            dpg.add_mouse_release_handler(callback=self.callback_mouse_release)
            # dpg.add_mouse_down_handler(callback=self.callback_mouse_down)
            dpg.add_mouse_move_handler(callback=self.callback_mouse_hover)
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=self.callback_mouse_left_click)
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=self.callback_mouse_right_click)
            dpg.add_key_press_handler(callback=self.callback_keypress)
        self.change_windows_size()

    @property
    def tree_node_now(self):
        return self._tree_node_now

    @tree_node_now.setter
    def tree_node_now(self, index):
        if dpg.does_item_exist(f"tree_{self.tree_node_now}"):
            dpg.bind_item_theme(f"tree_{self.tree_node_now}", self.default_theme)
        self._tree_node_now = index
        dpg.bind_item_theme(f"tree_{index}", self.choose_theme)

    def show_image(self):
        if len(self.update_viewer) == 0:
            return
        total = self.num_row * self.num_col
        image_index = self.image_index // total * total
        dpg.set_value('image_range', f'image: {image_index + 1} - {min(image_index + total, len(self.images))}')
        update_all = -1 in self.update_viewer
        image = np.zeros_like(self.viewer.data) if update_all else self.viewer.data
        alpha = 0.3
        for i in range(self.num_row):
            for j in range(self.num_col):
                k = image_index + i * self.num_col + j
                if k < len(self.images) and (update_all or k in self.update_viewer):
                    img = self.images[k]
                    masks = []
                    self.show_masks[k].zero_()
                    for idx in self.show_mask_ids[k]:
                        if idx > 0:
                            mask = self.tree2d[k].masks[idx - 1]
                            masks.append(mask)
                            self.show_masks[k][mask] = idx
                    if len(masks) > 0:
                        masks = get_colored_masks(masks, channels=img.shape[-1])
                        img = cv2.addWeighted(img, alpha, masks, 1. - alpha, 0)
                    choose = self.choices[k]
                    if choose > 0:
                        img = image_add_mask_boundary(img, self.tree2d[k].masks[choose - 1], kernel_size=9)
                    dx, dy, scale = self.image_crop[k]
                    img = cv2.warpAffine(img, np.array([[scale, 0, dx], [0, scale, dy]]), self.show_size)
                    y = i * (self.show_size[1] + self.padding) + self.padding // 2
                    x = j * (self.show_size[0] + self.padding) + self.padding // 2
                    image[y:y + self.show_size[1], x:x + self.show_size[0]] = img
        self.viewer.update(image, resize=True)
        self.update_viewer.clear()

    def change_windows_size(self):
        win_width = dpg.get_viewport_width()
        win_height = dpg.get_viewport_height()

        img_width = dpg.get_value('image_width')
        img_height = int(img_width * self.image_size[1] // self.image_size[0])
        for i in range(self.num_img):
            dx, dy, scale = self.image_crop[i]
            self.image_crop[i] = [dx, dy, scale * img_width / self.show_size[0]]
        self.show_size = (img_width, img_height)
        crl_width = dpg.get_item_width('control')
        num_column = (win_width - crl_width) // (img_width + self.padding)
        num_row = win_height // (img_height + self.padding)

        viewer_width = num_column * (img_width + self.padding)
        viewer_height = num_row * (img_height + self.padding)
        dpg.set_item_width(self.viewer.win_tag, viewer_width)
        dpg.set_item_height(self.viewer.win_tag, viewer_height)
        self.viewer.resize(viewer_width, viewer_height)

        dpg.set_item_height('control', win_height)
        dpg.set_item_pos('control', [win_width - crl_width - self.padding // 2, 0.])
        self.num_row = num_row
        self.num_col = num_column
        self.update_viewer.append(-1)

    def build_control_ui(self):
        # with dpg.collapsing_header():
        #     pass
        with dpg.group(horizontal=True):
            dpg.add_text('show width:')
            dpg.add_input_int(
                tag='image_width',
                default_value=256,
                min_value=100,
                width=40,
                step=0,
                callback=self.change_windows_size,
                min_clamped=True,
            )

            # dpg.add_button(label='0.8')
            # dpg.add_button(label='1.2')

            def reset_scale():
                for i in range(self.num_img):
                    self.image_crop[i] = [0, 0, self.show_size[0] / self.image_size[0]]
                self.update_viewer.append(-1)

            dpg.add_button(label='no-crop', callback=reset_scale)

        with dpg.group(horizontal=True):
            dpg.add_button(label='Load', width=50, height=30, callback=self.load_results)
            dpg.add_button(label='Save', width=50, height=30, callback=self.save_results)
        with dpg.group(horizontal=True):
            def last_image_page():
                total = self.num_col * self.num_row
                self.image_index -= total
                if self.image_index < 0:
                    self.image_index = len(self.images) // total * total
                self.update_viewer.append(-1)

            def next_image_page():
                self.image_index += self.num_col * self.num_row
                if self.image_index >= len(self.images):
                    self.image_index = 0
                self.update_viewer.append(-1)

            dpg.add_button(label='last', tag='last_page', callback=last_image_page)
            dpg.add_text(f'image: {self.image_index} - {self.image_index + self.num_col * self.num_row}',
                tag='image_range')
            dpg.add_button(label='next', tag='next_page', callback=next_image_page)
        with dpg.group(horizontal=True):
            dpg.add_button(label='insert', tag='tree_insert', callback=self.tree_insert)
            dpg.add_button(label='delete', tag='tree_delete', callback=self.tree_delete)
            dpg.add_button(label='arrange', callback=self.tree_arrange)
        with dpg.group(horizontal=True):  # show level
            def callback_show_level(level):
                def show_level():
                    for k in range(self.num_img):
                        self.show_mask_ids = []
                        for t in self.tree2d:
                            levels = t.get_levels()
                            if len(levels) > level:
                                self.show_mask_ids.append(levels[level].tolist())
                            else:
                                self.show_mask_ids.append([])
                    self.choices[:] = 0
                    self.update_viewer.append(-1)
                    return

                return show_level

            dpg.add_text('Level')
            max_level = max(tree2d.max_depth for tree2d in self.tree2d)
            for i in range(max_level):
                dpg.add_button(label=f"{i}",
                    tag=f'show_level_{i}',
                    callback=callback_show_level(i),
                    width=30,
                    height=30)
        dpg.add_separator()
        dpg.add_collapsing_header(label='root', tag='tree_0', default_open=True, open_on_arrow=True)
        dpg.bind_item_theme('tree_0', self.choose_theme)

    def tree_insert(self):
        now = self.tree_node_num
        self.tree[now] = {
            'parent': self.tree_node_now,
            'masks': np.zeros(self.num_img, dtype=np.int32),
            'depth': self.tree[self.tree_node_now]['depth'] + 1,
            'children': [],
        }
        self.tree[self.tree_node_now]['children'].append(now)
        dpg.add_collapsing_header(
            label=f'{now}',
            tag=f'tree_{now}',
            parent=f"tree_{self.tree_node_now}",
            open_on_arrow=True,
            indent=self.tree[now]['depth'] * 10,
            default_open=True,
        )
        self.tree_node_now = now
        self.tree_node_num += 1
        print(f"Insert Tree Node {now}")

    def tree_delete(self):
        now = self.tree_node_now
        if now == 0:
            return
        print(f"Delete Tree Node {now}")
        p = self.tree[now]['parent']

        def update_depth(x):
            for y in self.tree[x]['children']:
                self.tree[y]['depth'] = self.tree[x]['depth'] + 1
                dpg.configure_item(f"tree_{y}", indent=10 * self.tree[y]['depth'])
                update_depth(y)

        for c in self.tree[now]['children']:
            dpg.move_item(f"tree_{c}", parent=f"tree_{p}")
            self.tree[c]['parent'] = p
            self.tree[p]['children'].append(c)
        update_depth(p)
        self.tree[p]['children'].remove(now)
        self.tree_node_now = p
        dpg.delete_item(f"tree_{now}")
        del self.tree[now]

    def tree_arrange(self):
        print('Tree Arrange')
        now_level = [0]
        mapping = np.zeros(self.tree_node_num, dtype=np.int32)
        for x in range(1, self.tree_node_num):
            if dpg.does_item_exist(f"tree_{x}"):
                dpg.delete_item(f"tree_{x}")
        self.tree_node_num = 1
        depth = 0
        while len(now_level) > 0:
            next_level = []
            depth += 1
            for x in now_level:
                for c in self.tree[x]['children']:
                    next_level.append(c)
                    now = mapping[c] = self.tree_node_num
                    dpg.add_collapsing_header(
                        label=f"{now}",
                        tag=f"tree_{now}",
                        parent=f"tree_{mapping[x]}",
                        open_on_arrow=True,
                        indent=depth * 10,
                        default_open=True,
                    )
                    self.tree_node_num += 1
                    dpg.bind_item_theme(f"tree_{now}", self.default_theme)

            now_level = next_level
        new_tree = {}
        for k, v in self.tree.items():
            v['parent'] = mapping[v['parent']]
            v['children'] = [mapping[c] for c in v['children']]
            new_tree[mapping[k]] = v
        self.tree_node_now = mapping[self.tree_node_now]
        self.tree = new_tree

    def tree_node_callback(self):
        pass

    def run(self):
        dpg.setup_dearpygui()
        dpg.show_viewport()
        # dpg.set_primary_window('Primary Window', True)
        # dpg.start_dearpygui()
        # last_size = None
        while dpg.is_dearpygui_running():
            self.show_image()
            dpg.render_dearpygui_frame()
        #     # dpg.set_value('fps', f"FPS: {dpg.get_frame_rate()}")

    def focus_on_mask(self, image_idx, mask: Tensor):
        """聚焦给定mask"""
        non_zero = torch.nonzero(mask)
        if len(non_zero) == 0:
            return
        min_y, max_y = torch.aminmax(non_zero[:, 0])
        min_x, max_x = torch.aminmax(non_zero[:, 1])
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        width = max(min((max_x - min_x).item() * 1.2, self.image_size[0]), 1)
        height = max(min((max_y - min_y).item() * 1.2, self.image_size[1]), 1.)
        scale = min(self.show_size[0] / width, self.show_size[1] / height)
        dx = self.show_size[0] * 0.5 - cx * scale
        dy = self.show_size[1] * 0.5 - cy * scale
        self.image_crop[image_idx] = [dx.item(), dy.item(), scale]

    def get_image_pos_at_mouse(self) -> Tuple[int, float, float]:
        """得到鼠标指针对应的图片id及坐标"""
        if not dpg.is_item_hovered(self.viewer.image_tag):
            return -1, 0, 0
        x, y = self.viewer.get_mouse_pos()
        i = (x - self.padding // 2) // (self.show_size[0] + self.padding)
        j = (y - self.padding // 2) // (self.show_size[1] + self.padding)
        total = self.num_row * self.num_col
        image_idx = self.image_index // total * total + j * self.num_col + i
        if image_idx >= self.num_img:
            return -1, 0, 0
        h = y - (j * (self.show_size[1] + self.padding) + self.padding // 2)
        w = x - (i * (self.show_size[0] + self.padding) + self.padding // 2)
        if 0 <= w < self.show_size[0] and 0 <= h < self.show_size[1]:
            dx, dy, scale = self.image_crop[image_idx]
            w = (w - dx) / scale
            h = (h - dy) / scale
            return image_idx, w, h
        else:
            return -1, 0, 0

    def callback_mouse_hover(self):
        pass

    def callback_mouse_wheel(self, sender, app_data):
        if dpg.is_item_hovered(self.viewer.image_tag):
            image_idx, w, h = self.get_image_pos_at_mouse()
            if image_idx >= 0:
                speed = math.exp(0.25 * app_data)
                dx, dy, scale = self.image_crop[image_idx]
                dx = -w * scale * speed + w * scale + dx
                dy = -h * scale * speed + h * scale + dy
                self.image_crop[image_idx] = [dx, dy, speed * scale]
                self.update_viewer.append(image_idx)

    def callback_mouse_drag(self, sender, app_data):
        if dpg.is_item_hovered(self.viewer.image_tag) and app_data[0] == dpg.mvMouseButton_Left:
            image_idx, w, h = self.get_image_pos_at_mouse()
            if image_idx >= 0:
                dx, dy, scale = self.image_crop[image_idx]
                now_mouse_pos = self.viewer.get_mouse_pos()
                if self._last_mouse_pos is None:
                    self._last_mouse_pos = [now_mouse_pos[0] - app_data[1], now_mouse_pos[1] - app_data[2]]
                dx += now_mouse_pos[0] - self._last_mouse_pos[0]
                dy += now_mouse_pos[1] - self._last_mouse_pos[1]
                self.image_crop[image_idx] = [dx, dy, scale]
                self._last_mouse_pos = now_mouse_pos
                self.update_viewer.append(image_idx)

    def callback_mouse_down(self, sender, app_data):
        print('mouse down:', app_data)
        self._last_mouse_pos = self.viewer.get_mouse_pos()

    def callback_mouse_release(self, sender, app_data):
        # print('mouse release:', app_data)
        self._last_mouse_pos = None

    def callback_mouse_left_click(self):
        if dpg.is_item_left_clicked(self.viewer.image_tag):
            image_idx, w, h = self.get_image_pos_at_mouse()
            if image_idx >= 0:
                mask_idx = self.show_masks[image_idx][int(round(h)), int(round(w))].item()
                if mask_idx == self.choices[image_idx] or mask_idx == 0:
                    self.choices[image_idx] = 0
                else:
                    self.choices[image_idx] = mask_idx
                    self.focus_on_mask(image_idx, self.tree2d[image_idx].masks[mask_idx - 1])
                print(f'left click[{image_idx}], {h:.1f}, {w:.1f}: {self.choices[image_idx]} ')
                self.update_viewer.append(image_idx)
        for i in range(self.tree_node_num):
            if dpg.does_item_exist(f"tree_{i}") and dpg.is_item_left_clicked(f"tree_{i}"):
                dpg.bind_item_theme(f"tree_{self.tree_node_now}", self.default_theme)
                self.tree_node_now = i
                dpg.bind_item_theme(f"tree_{self.tree_node_now}", self.choose_theme)
                self.choices = self.tree[self.tree_node_now]['masks'].copy()
                for j, c in enumerate(self.choices):
                    k = i
                    while c == 0 and k > 0:
                        k = self.tree[k]['parent']
                        c = self.tree[k]['masks'][j]
                    if c > 0:
                        self.focus_on_mask(j, self.tree2d[j].masks[c - 1])
                    self.show_mask_ids[j] = self.tree2d[j].get_children(c)
                self.update_viewer.append(-1)

    def callback_mouse_right_click(self):
        if dpg.is_item_right_clicked(self.viewer.image_tag):
            image_idx, w, h = self.get_image_pos_at_mouse()
            if image_idx >= 0:
                if len(self.show_mask_ids[image_idx]) > 0:
                    mask_idx = self.show_masks[image_idx][int(round(h)), int(round(w))].item()
                else:
                    mask_idx = 0
                    for c in self.tree2d[image_idx].get_children(0):
                        if self.tree2d[image_idx].masks[c - 1, int(round(h)), int(round(w))]:
                            mask_idx = c
                print(f"right click image [{image_idx}, {h:.1f}, {w:.1f}], mask={mask_idx}")
                if mask_idx == 0:
                    return
                if dpg.is_key_down(dpg.mvKey_Control):
                    p = self.tree2d[image_idx].parent[mask_idx].item()
                    for c in self.tree2d[image_idx].get_children(p):
                        if c in self.show_mask_ids[image_idx]:
                            self.show_mask_ids[image_idx].remove(c)
                    self.show_mask_ids[image_idx].append(p)
                else:
                    added = False
                    for c in self.tree2d[image_idx].get_children(mask_idx):
                        self.show_mask_ids[image_idx].append(c)
                        added = True
                    if added and mask_idx in self.show_masks[image_idx]:
                        self.show_mask_ids[image_idx].remove(mask_idx)
                self.update_viewer.append(image_idx)

    def callback_keypress(self, sender, app_data):
        if dpg.is_key_pressed(dpg.mvKey_Prior) or dpg.is_key_pressed(dpg.mvKey_Up):
            dpg.get_item_callback('last_page')()
        elif dpg.is_key_pressed(dpg.mvKey_Next) or dpg.is_key_pressed(dpg.mvKey_Down) or \
            dpg.is_key_pressed(dpg.mvKey_Tab):
            dpg.get_item_callback('next_page')()
        elif dpg.is_key_pressed(dpg.mvKey_Insert):
            dpg.get_item_callback('tree_insert')()
        elif dpg.is_key_pressed(dpg.mvKey_Delete):
            dpg.get_item_callback('tree_delete')()
        elif dpg.is_key_pressed(dpg.mvKey_A):
            print(f"Add all choosen masks into tree node {self.tree_node_now}")
            self.tree[self.tree_node_now]['masks'] = self.choices.copy()
        elif dpg.is_key_pressed(dpg.mvKey_NumPad0):
            if dpg.does_item_exist('show_level_0'):
                dpg.get_item_callback('show_level_0')()
        elif dpg.is_key_pressed(dpg.mvKey_NumPad1):
            if dpg.does_item_exist('show_level_1'):
                dpg.get_item_callback('show_level_1')()
        elif dpg.is_key_pressed(dpg.mvKey_NumPad2):
            if dpg.does_item_exist('show_level_2'):
                dpg.get_item_callback('show_level_2')()
        elif dpg.is_key_pressed(dpg.mvKey_NumPad3):
            if dpg.does_item_exist('show_level_3'):
                dpg.get_item_callback('show_level_3')()
        elif dpg.is_key_pressed(dpg.mvKey_NumPad4):
            if dpg.does_item_exist('show_level_4'):
                dpg.get_item_callback('show_level_4')()
        elif dpg.is_key_pressed(dpg.mvKey_NumPad5):
            if dpg.does_item_exist('show_level_5'):
                dpg.get_item_callback('show_level_5')()
        elif dpg.is_key_pressed(dpg.mvKey_NumPad6):
            if dpg.does_item_exist('show_level_6'):
                dpg.get_item_callback('show_level_6')()
        elif dpg.is_key_pressed(dpg.mvKey_NumPad7):
            if dpg.does_item_exist('show_level_7'):
                dpg.get_item_callback('show_level_7')()
        elif dpg.is_key_pressed(dpg.mvKey_NumPad8):
            if dpg.does_item_exist('show_level_8'):
                dpg.get_item_callback('show_level_8')()
        elif dpg.is_key_pressed(dpg.mvKey_NumPad9):
            if dpg.does_item_exist('show_level_9'):
                dpg.get_item_callback('show_level_9')()
        elif dpg.is_key_pressed(dpg.mvKey_H):
            print('H is pressed')

    def save_results(self):
        torch.save(self.tree, self.cache_dir.joinpath('temp.pth'))
        print(f'Save tree to {self.cache_dir.joinpath("temp.pth")}')

    def load_results(self):
        if self.cache_dir.joinpath('temp.pth').exists():
            self.tree = torch.load(self.cache_dir.joinpath('temp.pth'), map_location='cpu')
            print(f"Load tree from {self.cache_dir.joinpath('temp.pth')}")
            self.tree_node_num = max(self.tree_node_num, max(self.tree.keys()) + 1)
            self.tree_node_now = 0
            self.tree_arrange()


if __name__ == '__main__':
    LabelGUI().run()
