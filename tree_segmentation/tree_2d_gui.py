from pathlib import Path
from typing import Dict, List, Union, Optional

import cv2
import dearpygui.dearpygui as dpg
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from extension import utils
from extension.utils import ImageViewer
from segment_anything.build_sam import Sam, build_sam
import tree_segmentation
from tree_segmentation.tree_2d_segmentation import MaskData, TreeData
from tree_segmentation.predictor import TreePredictor


def color_mask(mask: np.ndarray, max_value=None):
    if max_value is None:
        max_value = mask.max()
    cmap = matplotlib.colormaps['viridis'].resampled(max_value + 2)
    random_order = np.arange(max_value + 2)
    random_order[1:] = np.random.permutation(random_order[1:])
    mask = random_order[mask]
    mask_image = cmap(mask)
    mask_image = np.where(mask[:, :, None] == 0, 1., mask_image)
    return mask_image[..., :3].astype(np.float32)


def image_add_points(image: np.ndarray, points: np.ndarray, s=5, label: np.ndarray = None):
    if points.max() <= 1.1:
        points = points * np.array([image.shape[1], image.shape[0]])
    image = image.copy()
    H, W = image.shape[:2]
    for i in range(points.shape[0]):
        x, y = points[i]
        if label is None or label[i] == 1:
            c = np.array((1., 0, 0))
        else:
            c = np.array((0., 1., 0))
        for dx in range(-s, s + 1):
            for dy in range(-s, s + 1):
                if dx * dx + dy * dy <= s * s and 0 <= x + dx < W and 0 <= y + dy < H:
                    image[int(y + dy), int(x + dx), :] = c
    return image


def image_add_mask_boundary(image: np.ndarray, mask: Tensor, color=(1., 0, 0)):
    mask = F.interpolate(mask[None, None, :, :].float(), size=image.shape[:2], mode='nearest')
    boundary = F.avg_pool2d(mask, 7, 1, 3)
    boundary = torch.logical_and(boundary.ne(mask), mask).cpu().numpy()[0, 0]
    image = image.copy()
    image[boundary, :3] = color
    return image


class Tree2DGUI:
    def __init__(self, sam=None):
        self.device = torch.device('cuda')
        self._sam: Optional[Sam] = sam

        self.image_dir = Path('../segment_anything')
        self.image_paths = []
        self.image_index = 0

        self._predictor: Optional[TreePredictor] = None

        self._tree_2d: Optional[TreeData] = None
        self._2d_aux_data: Optional[Dict[Union[int, str], Tensor]] = None
        self._image: Optional[np.ndarray] = None
        self._points: Optional[np.ndarray] = None
        self._mask_data: Optional[MaskData] = None  # The results of one sample
        self._2d_levels: List[Tensor] = []
        self._2d_mask = None
        self.now_level = 0
        self._choose_mask = None
        self._refine_points = []
        self._refine_mask = None

        dpg.create_context()
        if not dpg.is_viewport_ok():
            dpg.create_viewport(title='2D Tree Label', width=1024 + 512, height=1024, x_pos=0, y_pos=256)

        with dpg.theme() as self.choose_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Header, (150, 100, 100), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Button, (150, 100, 100), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core)
        with dpg.theme() as self.default_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (50, 50, 50), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Header, (50, 50, 50), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core)

        self.viewer = ImageViewer(size=(512, 512), pos=(0, 0), no_move=True, no_resize=True)
        self.viewer.enable_dynamic_change()
        with dpg.window(pos=(1024, 0), width=512, no_close=True, height=256):
            self.make_control()
        with dpg.window(pos=(1024, 256), width=512, no_close=True, height=1024 - 256, label='tree'):
            self.make_show_tree_2d_mask_level()
            dpg.add_collapsing_header(label='root', tag='tree_0')
        with dpg.handler_registry():
            # dpg.add_mouse_drag_handler(callback=self.view_3d.callback_mouse_drag)
            # dpg.add_mouse_wheel_handler(callback=self.view_3d.callback_mouse_wheel)
            # dpg.add_mouse_release_handler(callback=self.view_3d.callback_mouse_release)
            dpg.add_mouse_move_handler(callback=self.callback_mouse_hover)
            dpg.add_mouse_click_handler(callback=self.callback_mouse_click)
            dpg.add_key_press_handler(callback=self.callback_keypress)
        # dpg.bind_item_theme('tree_0', self.default_theme)
        self.change_image_dir(None, {'file_path_name': '/home/wan/Pictures'})

    def change_image_dir(self, sender, app_data):
        self.image_dir = Path(app_data['file_path_name'])
        self.image_paths = []
        for path in self.image_dir.glob('*.*'):
            if path.suffix.lower() in utils.image_extensions:
                self.image_paths.append(path.name)
        self.image_index = 0
        print(f'GUI chage work dir to: {self.image_dir}, there are {len(self.image_paths)} images')
        self.change_image(index=0)

    def change_image(self, *, index=None, next=False, last=False):
        n = len(self.image_paths)
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
        self.image = utils.load_image(self.image_dir.joinpath(self.image_paths[self.image_index]))
        text = self.image_paths[self.image_index]
        if len(text) >= 36:
            text = '...' + text[-33:]
        dpg.set_value('image_path', f"{text:<36} ({self.image_index}/{n})")
        print(f'[GUI] load image from: {self.image_dir.joinpath(self.image_paths[self.image_index])}')

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, img_, min_size=1024, max_size=1024):
        if img_.ndim == 2:
            img_ = np.repeat(img_[:, :, None], 3, -1)
        elif img_.shape[-1] == 4:
            img_ = cv2.cvtColor(img_, cv2.COLOR_RGBA2RGB)
        h, w = img_.shape[:2]
        if w < min_size and h < min_size:
            scale = min(min_size / w, min_size / h)
            w, h = int(w * scale), int(h * scale)
            img_ = cv2.resize(img_, dsize=(w, h))
        elif w > max_size or h > max_size:
            scale = min(max_size / w, max_size / h)
            w, h = int(w * scale), int(h * scale)
            img_ = cv2.resize(img_, dsize=(w, h))

        if img_.dtype == np.uint8:
            img_ = img_.astype(np.float32) / 255.
        print(f'image shape: {img_.shape}, dtype: {img_.dtype}')
        self._image = img_
        self.viewer.update(img_, resize=False)

    def not_show_tree(self, levels):
        for nodes in reversed(levels):
            for x in nodes:
                x = x.item()
                if x > 0:
                    dpg.delete_item(f"tree_{x}")

    def show_tree(self):
        levels = self.levels_2d
        print(levels)
        for level, nodes in enumerate(levels):
            if level == 0:
                continue
            for x in nodes:
                x = x.item()
                p = self.tree2d.parent[x].item()
                # if self.tree2d.first[x].item() <= 0:
                #     dpg.add_text(f'{x}', tag=f"tree_{x}", parent=f"tree_{p}", indent=level * 10)
                # else:
                dpg.add_collapsing_header(label=f'{x}',
                    tag=f"tree_{x}",
                    parent=f"tree_{p}",
                    indent=level * 10,
                    leaf=self.tree2d.first[x].item() <= 0,
                    open_on_arrow=True,
                )
                dpg.bind_item_theme(f"tree_{x}", self.default_theme)
        return

    @property
    def sam(self) -> Sam:
        if self._sam is None:
            self.load_sam()
        return self._sam

    def load_sam(self):
        print('Loading SAM model...')
        sam_pth = Path('~/models/sam_vit_h_4b8939.pth').expanduser()
        self._sam = build_sam(sam_pth).to(self.device)
        # predictor = SamPredictor(sam)
        print('Loaded Sam')

    def get_value(self, name, default=None):
        if dpg.does_alias_exist(name):
            return dpg.get_value(name)
        else:
            print(f'[GUI] option {name} not in dpg')
            return default

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points):
        self._points = points
        if points is None or self.get_value('show_points', True):
            self.viewer.update(self.image, resize=True)
        else:
            self.viewer.update(image_add_points(self.image, points), resize=True)
        dpg.configure_item(f"level0", show=True)

    @property
    def mask_data(self):
        return self._mask_data

    @mask_data.setter
    def mask_data(self, data: MaskData = None):
        if data is None:
            dpg.hide_item('level0')
        else:
            dpg.show_item('level0')
        self._mask_data = [] if data is None else data

    @property
    def levels_2d(self):
        return self._2d_levels

    @levels_2d.setter
    def levels_2d(self, levels):
        self.choose_mask = None
        max_levels = len(levels)
        for i in range(1, 10):
            dpg.configure_item(f"level{i}", show=i < max_levels)
        self.not_show_tree(self._2d_levels)
        self._2d_levels = levels
        self.show_tree()

    @property
    def predictor(self) -> TreePredictor:
        if self._predictor is None:
            self._predictor = tree_segmentation.TreePredictor(
                self.sam,
                points_per_batch=self.get_value('points_per_batch', 64),
                pred_iou_thresh=self.get_value('pred_iou_thresh', 0.88),
                stability_score_thresh=self.get_value('stability_score_thresh', 0.95),
                box_nms_thresh=self.get_value('box_nms_thresh', 0.7),
            )
        if not self._predictor.is_image_set and self._image is not None:
            self._predictor.set_image(np.clip(self._image * 255., 0, 255).astype(np.uint8))
        return self._predictor

    @property
    def tree2d(self) -> TreeData:
        if self._tree_2d is None:
            self._tree_2d = TreeData(
                in_threshold=self.get_value('in_threshold'),
                in_thres_area=self.get_value('in_area_threshold'),
                union_threshold=self.get_value('union_threshold'),
                min_area=self.get_value('min_area'),
                device=self.device
            )
        return self._tree_2d

    @property
    def choose_mask(self):
        return self._choose_mask

    @choose_mask.setter
    def choose_mask(self, index=None):
        if index == self._choose_mask:
            index = None
        if self._choose_mask is not None:
            dpg.bind_item_theme(f"tree_{self._choose_mask}", self.default_theme)
        self._choose_mask = index
        if self._choose_mask is not None:
            dpg.bind_item_theme(f"tree_{self._choose_mask}", self.choose_theme)

    def show_masks(self):
        image = self.image
        # show_points
        if self.now_level in [0, -2] and self._points is not None and self.get_value('show_points'):
            image = image_add_points(image, self._points)
        # show level masks

        masks = None
        if self.now_level == 0:
            if not self.mask_data:
                masks = self.mask_data['masks']
        elif 0 < self.now_level < len(self.levels_2d):
            mask_index = self.levels_2d[self.now_level] - 1
            if mask_index.numel() > 0:
                assert 0 <= mask_index.min() and mask_index.max() < len(self.tree2d.data['masks'])
                masks = self.tree2d.data['masks'][mask_index]
        if masks is not None and masks.numel() > 0:
            self._2d_mask = masks * torch.arange(1, 1 + masks.shape[0], device=masks.device)[:, None, None]
            self._2d_mask = torch.amax(self._2d_mask, dim=0).int().cpu().numpy()
            image = color_mask(self._2d_mask, masks.shape[0])
            alpha = dpg.get_value('alpha')
            image = cv2.addWeighted(self.image, alpha, image[..., :3], 1 - alpha, 0)
        else:
            self._2d_mask = None
        # show_choose
        if self.choose_mask is not None and self.now_level != 0:
            mask = self.tree2d.data['masks'][self.choose_mask - 1]
            image = image_add_mask_boundary(image, mask, color=(0, 1., 0.))
        # refine:
        if self.now_level == -2 and len(self._refine_points) > 0:
            points = np.array(self._refine_points)
            image = image_add_points(image, points[:, :2], label=points[:, 2])
            if self._refine_mask is not None:
                image = image_add_mask_boundary(image, self._refine_mask[0][0], color=(0, 0, 1.))
        self.viewer.update(image, resize=False)

    def set_level(self, level):
        dpg.bind_item_theme(f"level{self.now_level}", self.default_theme)
        self.now_level = level
        dpg.bind_item_theme(f"level{self.now_level}", self.choose_theme)
        self._refine_points.clear()
        self._refine_mask = None
        if level == 0:
            print('show last run')
        elif level == -1:
            print('show all')
        elif level == -2:
            print('modify')
        else:
            print(f'show level: {level}')
        self.show_masks()

    def make_show_tree_2d_mask_level(self):
        def show_level_callback(level):
            def wrapper():
                self.set_level(level)

            return wrapper

        size = 30
        pad = 5
        # dpg.push_container_stack(self.viewer.win_tag)
        with dpg.group(horizontal=True):
            dpg.add_button(label='A', width=size, height=size, tag='level-1', callback=show_level_callback(-1))
            dpg.add_button(label='M', width=size, height=size, tag='level-2', callback=show_level_callback(-2))
            for tag in ['level-1', 'level-2']:
                dpg.bind_item_theme(tag, self.default_theme)
            for i in range(10):
                dpg.add_button(
                    label='T' if i == 0 else f"{i}",
                    width=size,
                    height=size,
                    tag=f"level{i}",
                    show=False,
                    callback=show_level_callback(i),
                    # pos=(pad + (size + pad) * i, 25),
                )
                dpg.bind_item_theme(f"level{i}", self.default_theme)
        # dpg.pop_container_stack()

    def make_control(self):
        with dpg.file_dialog(
            directory_selector=True,
            show=False,
            callback=self.change_image_dir,
            id="file_dialog_id",
            width=700,
            height=400,
            default_path='/home/wan/Pictures',
        ):
            dpg.add_file_extension("Image{.png,.jpg}", custom_text='[Image]')
        with dpg.group(horizontal=True):
            dpg.add_button(label='change dir', callback=lambda: dpg.show_item("file_dialog_id"))
            dpg.add_button(label='last', callback=lambda: self.change_image(last=True))
            dpg.add_text(label='no image', tag='image_path')
            dpg.add_button(label='next', callback=lambda: self.change_image(next=True))

        with dpg.group(horizontal=True):
            dpg.add_button(label='save', callback=self.save_result)
            dpg.add_button(label='load', callback=self.load_result)
            dpg.add_text('alpha')
            dpg.add_slider_float(min_value=0,
                max_value=1.,
                default_value=0.5,
                tag='alpha',
                callback=self.update_viewer)

        with dpg.group(horizontal=True):
            dpg.add_text('Threshold: iou=')
            o21 = dpg.add_input_float(tag='pred_iou_thresh', default_value=0.88, width=80, step=0)
            dpg.add_text('stability=')
            o22 = dpg.add_input_float(tag='stability_score_thresh', default_value=0.95, width=80, step=0)
            dpg.add_text('nms=')
            o23 = dpg.add_input_float(tag='box_nms_thresh', default_value=0.7, width=80, step=0)

        with dpg.group(horizontal=True):
            t1 = dpg.add_text('Merge: in=')
            o31 = dpg.add_input_float(default_value=0.9, step=0, tag='in_threshold')
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
            dpg.add_text('show')
            dpg.add_checkbox(tag='show_points')

        with dpg.group(horizontal=True):
            W, H = 70, 40
            dpg.add_button(width=W, height=H, label='Reset', callback=self.reset_2d)
            # dpg.add_button(width=W, height=H, label='New Pose', callback=self.new_camera_pose)
            dpg.add_button(width=W, height=H, label='Stage1', callback=self.run_tree_seg_2d_stage1)
            dpg.add_button(width=W, height=H, label='Stage2', callback=self.run_tree_seg_2d_stage2)
            dpg.add_button(width=W, height=H, label='Post', callback=None)
            dpg.add_button(width=W, height=H, label='AutoRun', callback=self.autorun_tree_seg_2d)
            dpg.add_button(width=W, height=H, label='Rebuild', callback=self.rebuild_tree)

        def changed_callback(item):
            def changed(*args):
                if self._predictor is not None:
                    setattr(self._predictor, item, dpg.get_value(item))
                    print(f'[Tree 2D] Change {item} to {getattr(self._predictor, item)}')

            return changed

        for item in [o11, o12, o21, o22, o23]:
            print(item)
            dpg.set_item_callback(item, changed_callback(item))

        with dpg.group(horizontal=True, pos=(10, 230)):
            dpg.add_text(tag='fps')
            dpg.add_text(tag='progress')
            dpg.add_button(label='Exit', callback=lambda: exit(1))

    def rebuild_tree(self):
        self.choose_mask = None
        self._refine_points.clear()
        self._refine_mask = None
        self.tree2d.reset(self.tree2d.data)
        self.tree2d.update_tree()
        self.tree2d.remove_not_in_tree()
        self.levels_2d = self.tree2d.get_levels()
        self.set_level(1)
        print(f'[GUI] rebuild tree')

    def reset_2d(self):
        self.now_level = 0
        print('[GUI] reset 2D')
        # self._tree2d = None
        self._tree_2d = None
        self.mask_data = None
        self._2d_mask = None
        self._points = None
        self._2d_aux_data = None
        self.levels_2d = []
        self._refine_mask = None
        self._refine_points.clear()
        self.choose_mask = None
        if self._image is None:
            self.viewer.data[:] = 255
        else:
            self.viewer.update(self._image, False)

    def run_tree_seg_2d_stage1(self):
        if not self.mask_data:
            self.reset_2d()
        image = self.image
        points = self.tree2d.sample_grid(dpg.get_value('points_per_side'))
        self._points = points
        self.viewer.update(image_add_points(image, points), resize=False)
        if not self.predictor.is_image_set:
            self.predictor.set_image(np.clip(self.image * 255, 0, 255).astype(np.uint8))
        self.mask_data = self.predictor.process_points(points)
        dpg.get_item_callback('level0')()
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
        points = self.tree2d.sample_by_counts(dpg.get_value('points_per_update'))
        self._points = points
        if points is None:
            self.viewer.update(self.image, resize=True)
            # print('ERROR: no unfilled mask')
            print(f'Update complete')
            return False
        self.viewer.update(image_add_points(self.image, points), resize=False)
        if not self.predictor.is_image_set:
            self.predictor.set_image(np.clip(self.image * 255, 0, 255).astype(np.uint8))
        self.mask_data = self.predictor.process_points(points)
        dpg.get_item_callback('level0')()
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

    def update_viewer(self):
        dpg.get_item_callback(f"level{self.now_level}")()

    def run(self):
        dpg.setup_dearpygui()
        dpg.show_viewport()
        # dpg.set_primary_window('Primary Window', True)
        # dpg.start_dearpygui()
        last_size = None
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
            # now_size = dpg.get_item_width(view_3d._win_id), dpg.get_item_height(view_3d._win_id)
            # if last_size != now_size:
            #     dpg.configure_item('control', pos=(dpg.get_item_width(view_3d._win_id), 0))
            #     dpg.set_viewport_width(dpg.get_item_width(view_3d._win_id) + dpg.get_item_width('control'))
            #     dpg.set_viewport_height(dpg.get_item_height(view_3d._win_id))
            #     last_size = now_size
            # dpg.set_value('fps', f"FPS: {dpg.get_frame_rate()}")

    def run_refine(self):
        self._refine_mask = None
        self.show_masks()
        if len(self._refine_points) == 0:
            return
        points = np.array(self._refine_points, dtype=np.float64)
        masks, ious, _ = self.predictor.predict(points[:, :2],
            points[:, 2].astype(np.int32),
            multimask_output=False)
        self._refine_mask = (torch.from_numpy(masks).to(self.tree2d.device), float(ious[0]))
        print(utils.show_shape(self._refine_mask))
        self.show_masks()

    def callback_mouse_hover(self, sender, app_data):
        if self.now_level == -2:
            return
        if dpg.is_item_hovered(self.viewer.image_tag):
            x, y = self.viewer.get_mouse_pos()
            radiu = 5
            if self._2d_mask is not None:
                H1, W1 = self._2d_mask.shape
                W2, H2 = self.viewer.size
                x, y = int(x * W1 / W2), int(y * H1 / H2)
                mask_ids = self._2d_mask[max(0, y - radiu):y + radiu, max(0, x - radiu):x + radiu]
                mask_ids, counts = np.unique(mask_ids, return_counts=True)
                mask_id = mask_ids[np.argmax(counts)] - 1

                if mask_id < 0:
                    self.viewer.data = self.viewer.origin_data
                    return
                mask = None
                if self.now_level == 0:
                    if not self.mask_data:
                        mask = self.mask_data['masks'][mask_id]
                else:
                    mask = self.tree2d.data['masks'][self.levels_2d[self.now_level][mask_id] - 1]
                if mask is not None:
                    self.viewer.data = image_add_mask_boundary(self.viewer.origin_data, mask)
        else:
            if self.viewer._origin_data is not None:
                self.viewer.data = self.viewer._origin_data
                self.viewer._origin_data = None

    def callback_mouse_click(self, sender, app_data):
        if self.now_level == -2:  # modify
            x, y = self.viewer.get_mouse_pos()
            if dpg.is_item_left_clicked(self.viewer.image_tag):
                self._refine_points.append((x, y, 1))
            elif dpg.is_item_right_clicked(self.viewer.image_tag):
                self._refine_points.append((x, y, 0))
            self.run_refine()
            return

        if dpg.is_item_left_clicked(self.viewer.image_tag):
            if self._2d_mask is None:
                return
            x, y = self.viewer.get_mouse_pos()
            H1, W1 = self._2d_mask.shape
            W2, H2 = self.viewer.size
            x, y = int(x * W1 / W2), int(y * H1 / H2)
            mask_id = self._2d_mask[y, x].item() - 1
            if self.now_level > 0 and mask_id >= 0:
                mask_id = self.levels_2d[self.now_level][mask_id - 1].item()
            if mask_id > 0:
                print('click:', mask_id)
                self.choose_mask = mask_id + 1
                self.show_masks()
        for nodes in self.levels_2d:
            for x in nodes:
                x = x.item()
                if x != 0 and dpg.is_item_left_clicked(f"tree_{x}"):
                    print('click item', x)
                    self.choose_mask = x
                    self.show_masks()

    def callback_keypress(self, sender, app_data):
        print('sender:', sender, 'app_data:', app_data)
        if dpg.is_key_pressed(dpg.mvKey_M):
            self.set_level(-2)
        elif dpg.is_key_pressed(dpg.mvKey_1) or dpg.is_key_pressed(dpg.mvKey_NumPad1):
            self.set_level(1)
        elif dpg.is_key_pressed(dpg.mvKey_2) or dpg.is_key_pressed(dpg.mvKey_NumPad2):
            self.set_level(2)
        elif dpg.is_key_pressed(dpg.mvKey_3) or dpg.is_key_pressed(dpg.mvKey_NumPad3):
            self.set_level(3)
        elif dpg.is_key_pressed(dpg.mvKey_4) or dpg.is_key_pressed(dpg.mvKey_NumPad4):
            self.set_level(4)
        elif dpg.is_key_pressed(dpg.mvKey_5) or dpg.is_key_pressed(dpg.mvKey_NumPad5):
            self.set_level(5)
        elif dpg.is_key_pressed(dpg.mvKey_6) or dpg.is_key_pressed(dpg.mvKey_NumPad6):
            self.set_level(6)
        elif dpg.is_key_pressed(dpg.mvKey_7) or dpg.is_key_pressed(dpg.mvKey_NumPad7):
            self.set_level(7)
        elif dpg.is_key_pressed(dpg.mvKey_8) or dpg.is_key_pressed(dpg.mvKey_NumPad8):
            self.set_level(8)
        elif dpg.is_key_pressed(dpg.mvKey_9) or dpg.is_key_pressed(dpg.mvKey_NumPad9):
            self.set_level(9)
        elif dpg.is_key_pressed(dpg.mvKey_0) or dpg.is_key_pressed(dpg.mvKey_NumPad0):
            self.set_level(0)
        elif dpg.is_key_pressed(dpg.mvKey_Z):
            if len(self._refine_points):
                self._refine_points.pop(-1)
                self.run_refine()
        elif dpg.is_key_pressed(dpg.mvKey_Escape):
            self._refine_points.clear()
            self._refine_mask = None
        elif dpg.is_key_pressed(dpg.mvKey_D) or dpg.is_key_pressed(dpg.mvKey_Delete):
            if self.choose_mask is not None:
                self.tree2d.node_delete(self.choose_mask, move_children=True)
                self.levels_2d = self.tree2d.get_levels()
                self.show_masks()
                print(f'[GUI] Delete node {self.choose_mask} from tree')
        elif dpg.is_key_pressed(dpg.mvKey_A) or dpg.is_key_pressed(dpg.mvKey_Insert):
            if self._refine_mask is not None:
                iou_preds = torch.tensor([self._refine_mask[1]], device=self.tree2d.device)
                self.tree2d.cat(MaskData(masks=self._refine_mask[0], iou_preds=iou_preds))
                self.tree2d.update_tree()
                self.levels_2d = self.tree2d.get_levels()
                self.choose_mask = None if self.tree2d.parent[self.tree2d.cnt] == -1 else self.tree2d.cnt
                self.set_level(1)
                print(f'[GUI] Add new node to tree')
        elif dpg.is_key_pressed(dpg.mvKey_R):
            if self._refine_mask is not None and self.choose_mask is not None:
                self.tree2d.data['masks'][self.choose_mask - 1] = self._refine_mask[0][0]
                self.tree2d.data['iou_preds'][self.choose_mask - 1] = self._refine_mask[1]
                self.tree2d.data['area'][self.choose_mask - 1] = self._refine_mask[0][0].sum()
                self._refine_points.clear()
                self._refine_mask = None
                self.show_masks()
                print(f'[GUI] Refine node {self.choose_mask} in tree')
        elif dpg.is_key_pressed(dpg.mvKey_S):
            if self._refine_mask is not None and self.choose_mask is not None:
                self.tree2d.data['masks'][self.choose_mask - 1] &= torch.logical_not(self._refine_mask[0][0])
                self.tree2d.data['area'][self.choose_mask - 1] = self.tree2d.data['masks'][self.choose_mask - 1].sum()
                self._refine_points.clear()
                self._refine_mask = None
                self.show_masks()
                print(f'[GUI] Subtract mask from node {self.choose_mask} in tree')
        elif dpg.is_key_pressed(dpg.mvKey_F):
            if self._refine_mask is not None and self.choose_mask is not None:
                self.tree2d.data['masks'][self.choose_mask - 1] |= self._refine_mask[0][0]
                self.tree2d.data['area'][self.choose_mask - 1] = self.tree2d.data['masks'][self.choose_mask - 1].sum()
                self._refine_points.clear()
                self._refine_mask = None
                self.show_masks()
                print(f'[GUI] Fusion mask from node {self.choose_mask} in tree')

    def save_result(self):
        if self._tree_2d is not None:
            save_path = self.image_dir.joinpath(self.image_paths[self.image_index]).with_suffix('.tree2d')
            self._tree_2d.save(save_path)
            print(f'[GUI]: save tree segment results to {save_path}')

    def load_result(self):
        load_path = self.image_dir.joinpath(self.image_paths[self.image_index]).with_suffix('.tree2d')
        if load_path.exists():
            self.reset_2d()
            self.tree2d.load(load_path)
            self.levels_2d = self.tree2d.get_levels()
            print(f'[GUI]: load tree segment results from {load_path}')
            dpg.get_item_callback("level1")()
        else:
            print(f"[GUI]: {load_path} is not exist to load")


if __name__ == '__main__':
    Tree2DGUI().run()
