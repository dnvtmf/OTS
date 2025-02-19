from typing import Callable
from pathlib import Path
import math

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
import torch
from torch import Tensor

from .image_viewer import ImageViewer
from ..lazy_import import LazyImport
from tree_segmentation.extension import ops_3d


class Viewer3D(ImageViewer):

    def __init__(self, renderer: Callable, size=(100, 100), pad=0, tag='3d', no_resize=True, no_move=True, **kwargs):
        super().__init__(size=size, pad=pad, tag=tag, no_resize=no_resize, no_move=no_move, **kwargs)

        self.renderer = renderer
        self.fovy = math.radians(60.)
        self.Tv2s = ops_3d.camera_intrinsics(size=size, fovy=self.fovy)
        self.Ts2v = ops_3d.camera_intrinsics(size=size, fovy=self.fovy, inv=True)

        self.up = torch.tensor([0, 1., 0.])
        self.eye = torch.tensor([0., 0., 2.0])
        self.at = torch.tensor([0., 0., 0.])
        # self.
        self._last_mouse_pos = None
        self._last_mouse_idx = None
        self.rate_rotate = self.fovy / self.height  # 旋转速度
        self.rate_translate = 1.  # 平移速度
        self.need_update = True

    def resize(self, W: int = None, H: int = None, channels: int = None):
        if super().resize(W, H, channels):
            self.need_update = True

    def callback_mouse_down(self, sender, app_data):
        # if dpg.is_item_hovered(self._img_id):
        #     self._last_mouse_pos = self.get_mouse_pos()
        #     self._last_mouse_idx = app_data[0]
        #     print(sender, app_data, self._last_mouse_pos)
        # else:
        #     self._last_mouse_pos = None
        #     self._last_mouse_idx = None
        pass

    def callback_mouse_release(self, sender, app_data):
        self._last_mouse_pos = None
        self._last_mouse_idx = None

    def callback_mouse_wheel(self, sender, app_data):
        if not dpg.is_item_hovered(self._img_id):
            return
        self.scale(app_data)

    def callback_mouse_drag(self, sender, app_data):
        if not dpg.is_item_hovered(self._img_id):
            return
        if app_data[0] == dpg.mvMouseButton_Left:
            if self._last_mouse_pos is not None and self._last_mouse_idx == app_data[0]:
                now_pos = self.get_mouse_pos()
                self.rotate(now_pos[0] - self._last_mouse_pos[0], now_pos[1] - self._last_mouse_pos[1])
        elif app_data[0] == dpg.mvMouseButton_Right:
            if self._last_mouse_pos is not None and self._last_mouse_idx == app_data[0]:
                now_pos = self.get_mouse_pos()
                self.translate(now_pos[0] - self._last_mouse_pos[0], now_pos[1] - self._last_mouse_pos[1])
        self._last_mouse_pos = self.get_mouse_pos()
        self._last_mouse_idx = app_data[0]

    def rotate(self, dx: float, dy: float):
        if dx == 0 and dy == 0:
            return
        radiu = (self.eye - self.at).norm()
        dir_vec = ops_3d.normalize(self.eye - self.at)
        right_vec = ops_3d.normalize(torch.cross(self.up, dir_vec), dim=-1)
        theta = -dy * self.rate_rotate
        dir_vec = ops_3d.quaternion.xfm(dir_vec, ops_3d.quaternion.from_rotate(right_vec, right_vec.new_tensor(theta)))

        right_vec = ops_3d.normalize(torch.cross(self.up, dir_vec), dim=-1)
        up_vec = torch.cross(dir_vec, right_vec)
        theta = -dx * self.rate_rotate
        dir_vec = ops_3d.quaternion.xfm(dir_vec, ops_3d.quaternion.from_rotate(up_vec, up_vec.new_tensor(float(theta))))
        self.eye = self.at + ops_3d.normalize(dir_vec) * radiu
        self.up = up_vec
        self.need_update = True

    def translate(self, dx: float, dy: float):
        """在垂直于视线方向进行平移, 即在view space进行平移"""
        if dx == 0 and dy == 0:
            return
        Tw2v = ops_3d.look_at(self.eye, self.at, self.up)
        p1 = ops_3d.xfm(ops_3d.xfm(self.at, Tw2v), self.Tv2s)

        p2 = p1.clone()
        p2[0] += dx * p1[2]
        p2[1] += dy * p1[2]
        Tv2w = ops_3d.look_at(self.eye, self.at, self.up, inv=True)
        p1 = ops_3d.xfm(ops_3d.xfm(p1, self.Ts2v), Tv2w)
        p2 = ops_3d.xfm(ops_3d.xfm(p2, self.Ts2v), Tv2w)
        delta = (p1 - p2)[:3] * self.rate_translate
        self.at += delta
        self.eye += delta
        self.need_update = True

    def scale(self, delta=0.0):
        self.eye = self.at + (self.eye - self.at) * 1.1 ** (-delta)
        self.need_update = True

    def update(self, resize=False):
        if not self.need_update:
            return
        self.need_update = False
        Tw2v = ops_3d.look_at(self.eye, self.at, self.up)
        image = self.renderer(Tw2v, self.fovy, self.size)
        if isinstance(image, Tensor):
            image = image.detach().cpu().numpy()
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255
        image = image.astype(np.float32)
        if image.ndim == 4:
            image = image[0]
        if image.shape[-1] not in [3, 4]:
            assert image.shape[0] in [3, 4]
            image = image.transpose(1, 2, 0)
        if resize:
            image = cv2.resize(image, self.size)
        self.resize(image.shape[1], image.shape[0], image.shape[2])
        self.data = image

    def set_fovy(self, fovy=60.):
        self.fovy = math.radians(fovy)
        self.Tv2s = ops_3d.camera_intrinsics(size=self.size, fovy=self.fovy)
        self.Ts2v = ops_3d.camera_intrinsics(size=self.size, fovy=self.fovy, inv=True)
        self.need_update = True

    def set_pose(self, eye=None, at=None, up=None, Tw2v=None, Tv2w=None):
        if Tv2w is None and Tw2v is not None:
            Tv2w = Tw2v.inverse()
        if Tv2w is not None:
            Tv2w = Tv2w.view(-1, 4, 4)[0].to(self.eye.device)
            eye = Tv2w[:3, 3]
            at = eye - Tv2w[:3, 2]
            up = Tv2w[:3, 1]
        if eye is not None:
            self.eye = eye
        if at is not None:
            self.at = at
        if up is not None:
            self.up = up
        self.need_update = True

    def set_need_update(self, need_update=True):
        self.need_update = need_update


def simple_3d_viewer(rendering, size=(400, 400)):
    dpg.create_context()
    dpg.create_viewport(title='Custom Title')
    with dpg.window(tag='Primary Window'):
        img = Viewer3D(rendering, size=size, no_resize=False, no_move=True)
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


def test():
    import torch
    import nvdiffrast.torch as dr
    from tree_segmentation.extension import Mesh
    from ... import ops_3d
    mesh_path = Path('~/data/meshes/spot/spot.obj').expanduser()
    mesh = Mesh.load(mesh_path)
    mesh = mesh.unit_size()
    mesh = mesh.cuda()
    print(mesh)

    glctx = dr.RasterizeCudaContext()
    lgt = torch.randn(3).cuda()

    @torch.no_grad()
    def rendering(Tw2v, fovy, size):
        Tv2c = ops_3d.perspective(size=size, fovy=fovy).cuda()
        Tw2c = Tv2c @ Tw2v.to(Tv2c.device)
        pos = ops_3d.xfm(mesh.v_pos, Tw2c)[None]
        assert pos.ndim == 3
        tri = mesh.f_pos.int()
        resolution = min(2048, size[0] // 8 * 8), min(2048, size[1] // 8 * 8)
        rast, _ = dr.rasterize(glctx, pos, tri, resolution=resolution)
        nrm, _ = dr.interpolate(mesh.v_nrm, rast, tri)
        nrm = ops_3d.normalize(nrm)
        if mesh.v_tex is not None:
            uv, _ = dr.interpolate(mesh.v_tex[None], rast, mesh.f_tex.int())
            kd = dr.texture(mesh.material['kd'].data[..., :3].contiguous(), uv) if 'kd' in mesh.material else None
            ks = dr.texture(mesh.material['ks'].data, uv) if 'ks' in mesh.material else None
        else:
            kd, ks = None, None
        if kd is None:
            kd = nrm.new_ones((*nrm.shape[:-1], 3))
        if ks is None:
            ks = nrm.new_zeros((*nrm.shape[:-1], 3))
        img = ops_3d.HalfLambert(nrm, lgt, kd)
        # img = ops_3d.Blinn_Phong(nrm, lgt, ops_3d.normalize(Tv2c[..., 3, :3]), kd, ks)
        img = dr.antialias(img, rast, pos, tri)
        return img

    simple_3d_viewer(rendering)
