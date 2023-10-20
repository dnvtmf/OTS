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

from tree_segmentation.extension import utils
from tree_segmentation.extension import ops_3d
from tree_segmentation.extension.utils import ImageViewer, Viewer3D


class LabelGUI:
    def __init__(self):
        self.image_dir = Path('~/data/NeRF/D_NeRF/lego/train').expanduser()
        self.seg_2d_dir = Path('~/data/NeRF/D_NeRF/lego/tree_seg/train').expanduser()
        self.images = np.stack([utils.load_image(path) for path in self.image_dir.glob('*.png')])
        self.images = self.images.astype(np.float32) / 255.
        self.image_size = (self.images.shape[2], self.images.shape[1])
        print(f"Load images {self.images.shape} from {self.image_dir}")

        self.padding = 10

        dpg.create_context()
        # self.set_default_font()
        dpg.create_viewport(title='Tree Segmentation Labeling', min_width=400, resizable=True)
        dpg.get_viewport_width()
        self.image_index = 0
        self.show_size = [256, 256 * self.image_size[1] // self.image_size[0]]
        self.num_row = 1
        self.num_column = 1

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
        self.change_windows_size()

    def show_image(self):
        total = self.num_row * self.num_column
        self.image_index = self.image_index // total * total
        dpg.set_value('image_range',
            f'image: {self.image_index + 1} - {min(self.image_index + total, len(self.images))}')
        image = np.zeros_like(self.viewer.data)
        for i in range(self.num_row):
            for j in range(self.num_column):
                k = self.image_index + i * self.num_column + j
                if k < len(self.images):
                    img = cv2.resize(self.images[k], self.show_size, interpolation=cv2.INTER_LINEAR)
                    y = i * (self.show_size[1] + self.padding)
                    x = j * (self.show_size[0] + self.padding)
                    image[y:y + self.show_size[1], x:x + self.show_size[0]] = img
        self.viewer.update(image, resize=True)

    def change_windows_size(self):
        win_width = dpg.get_viewport_width()
        win_height = dpg.get_viewport_height()

        img_width = dpg.get_value('image_width')
        img_height = int(img_width * self.image_size[1] // self.image_size[0])
        self.show_size = (img_width, img_height)
        crl_width = dpg.get_item_width('control')
        num_column = (win_width - crl_width + self.padding) // (img_width + self.padding)
        num_row = (win_height + self.padding) // (img_height + self.padding)

        viewer_width = num_column * img_width + (num_column - 1) * self.padding
        viewer_height = num_row * img_height + (img_height - 1) * self.padding
        dpg.set_item_width(self.viewer.win_tag, viewer_width)
        dpg.set_item_height(self.viewer.win_tag, viewer_height)
        self.viewer.resize(viewer_width, viewer_height)

        dpg.set_item_height('control', win_height)
        dpg.set_item_pos('control', [win_width - crl_width - self.padding // 2, 0.])
        self.num_row = num_row
        self.num_column = num_column
        self.show_image()

    def build_control_ui(self):
        # with dpg.collapsing_header():
        #     pass
        dpg.add_input_int(
            label='image width',
            tag='image_width',
            default_value=256,
            min_value=100,
            width=100,
            step=0,
            callback=self.change_windows_size,
            min_clamped=True,
        )
        with dpg.group(horizontal=True):
            def last_image_page():
                total = self.num_column * self.num_row
                self.image_index -= total
                if self.image_index < 0:
                    self.image_index = len(self.images) // total * total
                self.show_image()

            def next_image_page():
                self.image_index += self.num_column * self.num_row
                if self.image_index >= len(self.images):
                    self.image_index = 0
                self.show_image()

            dpg.add_button(label='last', callback=last_image_page)
            dpg.add_text(f'image: {self.image_index} - {self.image_index + self.num_column * self.num_row}',
                tag='image_range')
            dpg.add_button(label='next', callback=next_image_page)

    def run(self):
        dpg.setup_dearpygui()
        dpg.show_viewport()
        # dpg.set_primary_window('Primary Window', True)
        dpg.start_dearpygui()
        # last_size = None
        # while dpg.is_dearpygui_running():
        #     dpg.render_dearpygui_frame()
        #     # dpg.set_value('fps', f"FPS: {dpg.get_frame_rate()}")


if __name__ == '__main__':
    LabelGUI().run()
