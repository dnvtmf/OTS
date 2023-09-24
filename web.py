import os
from pathlib import Path
from typing import Union

import cv2
import gradio as gr
import numpy as np
import seaborn as sns
import torch

from segment_anything.build_sam import Sam
from semantic_sam import SemanticSAM
from tree_segmentation import Tree3Dv2
from tree_segmentation.extension import utils
from tree_segmentation.tree_3d import TreeSegment
from tree_segmentation.util import get_colored_masks, search_folder, get_hash_name


class WebUI(TreeSegment):

    def __init__(self, args, model: Union[Sam, SemanticSAM, None] = None):
        super().__init__(args, model)
        self._model_type = 'Semantic-SAM-T'
        self.cfg = {}
        self.image_dir = Path(__file__).parent.joinpath('images').expanduser()
        self.image_index = 0
        self.image_paths = []

        self.mesh_dir = Path(__file__).parent.joinpath('meshes').expanduser()
        self.mesh_index = 0
        self.mesh_paths = []

        self.cache_web = self.cache_root.joinpath('web_cache')
        self.cache_web.mkdir(exist_ok=True)

        with gr.Blocks() as tree_seg_2d_block:
            self.build_tree_seg_2d_ui()

        with gr.Blocks() as tree_seg_3d_block:
            self.build_tree_seg_3d_ui()

        with gr.Blocks() as option_block:
            self.build_options()
        self.web_ui = gr.TabbedInterface([tree_seg_2d_block, tree_seg_3d_block, option_block],
            ["2D Tree Segmentation", "3D Tree Segmentation", "Options"])

    def run(self):
        self.web_ui.launch(share=False, server_name='0.0.0.0')

    def get_value(self, name, default=None):
        if name in self.cfg:
            return self.cfg[name]
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
            print(f'[Web] option {name} not in dpg')
            return default

    def set_value(self, key: str, default=None):
        if default is not None:
            self.cfg[key] = default

        def fn(value):
            self.cfg[key] = value
            print(f'[Web] set {key} to {value}')

        return fn

    def load_images(self):
        # self.gallery.visible = True
        self.image_paths = []
        for filenanme in os.listdir(self.image_dir):
            image_path = self.image_dir.joinpath(filenanme)
            if image_path.suffix in utils.image_extensions:
                self.image_paths.append(image_path)
        self.image_index = 0
        return self.image_paths

    def upload_images(self, files):
        self.image_index = len(self.image_paths)
        for file in files:
            self.image_paths.append(Path(file.name))
            print(f'[GUI] upload image: {self.image_paths[-1].name}')
        return self.image_paths

    def change_image(self, evt: gr.SelectData):
        self.image_index = evt.index
        image = utils.load_image(self.image_paths[self.image_index])
        H, W = image.shape[:2]
        max_size = self.get_value('max_image_size', 1024)
        min_size = self.get_value('min_image_size', 256)

        scale = min(min(max_size / H, max_size / W), max(max(min_size / H, min_size / W), 1.0))
        H2, W2 = int(H * scale), int(W * scale)
        print(f'[Web] change image shape: {W}x{H} --> {W2} x {H2}')
        image = cv2.resize(image, (W2, H2), interpolation=cv2.INTER_CUBIC)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        assert image.ndim == 3 and image.shape[-1] == 3
        self.image = image.astype(np.float32)[..., :3] / 255.
        return self.image

    def build_options(self):
        def add_option(name, default, **kwargs):
            x = gr.Number(default, label=name, interactive=True, **kwargs)
            x.change(self.set_value(name, default), x)

        with gr.Group():
            gr.Markdown('Segment Options')
            r_model_type = gr.Radio(["SAM", "SAM-L", "SAM-B", "Semantic-SAM-L", "Semantic-SAM-T"],
                label='Base Model',
                value=self._model_type)
            with gr.Row():
                add_option('points_per_batch', 64, precision=0)
                add_option('stability_score_thresh', 0.95)
                add_option('pred_iou_thresh', 0.88)
                add_option('box_nms_thresh', 0.7)

        def change_model(model_type):
            if self._model_type != model_type:
                self._model_type = model_type
                self._model = None
            print('[Web] Model type change to', model_type)

        r_model_type.change(change_model, r_model_type)

        with gr.Group():
            gr.Markdown('Tree Segmentation Options')
            with gr.Row():
                add_option('points_per_side', 32, precision=0)
                add_option('max_steps', 100, precision=0)
                add_option('points_per_update', 256, precision=0)
                add_option('min_area', 100, precision=0)
            with gr.Row():
                add_option('in_threshold', 0.9)
                add_option('in_area_threshold', 50, precision=0)
                add_option('union_threshold', 0.1)
                add_option('alpha', 0.3)
        # with gr.Group():
        #     gr.Markdown('Tree Segmentation 2D')

    def update_2d_results(self):
        data = []
        for i in range(len(self.img_levels)):
            if i + 1 < len(self.levels_2d):
                masks = self.tree2d.masks[self.levels_2d[i + 1] - 1]
                image = get_colored_masks(masks)
                alpha = self.get_value('alpha', 0.3)
                if self.image.dtype == np.uint8:
                    image = (image * 255).astype(np.uint8)
                image = cv2.addWeighted(self.image[..., :3], alpha, image[..., :3], 1 - alpha, 0)
                data.append(self.img_levels[i].update(image, visible=True, width=image.shape[1]))
            else:
                data.append(self.img_levels[i].update(value=None, visible=i == 0))
        return data

    def reset_2d(self):
        super().reset_2d()
        return self.update_2d_results()

    def autorun_tree_seg_2d(self, *args, progress=gr.Progress(), **kwargs):
        if self._image is None:
            print(f'[Web] autorun_tree_seg_2d no image')
            return
        progress(0, desc="Grid Sample Stage")
        super().run_tree_seg_2d_stage1()
        print('[Tree 2D]: autorun stage1')
        max_steps = self.get_value('max_steps', 100)
        progress(0.1, desc="Heuristic Sample Stage")
        for step in range(max_steps):
            if not super().run_tree_seg_2d_stage2():
                break
            progress((step + 2) / (max_steps + 2), desc="Heuristic Sample Stage")
            print(f'[Tree 2D]: autorun stage2 {step + 1}/{max_steps}')
        print(f"[Web] complete autorun_tree_seg_2d")
        progress((max_steps + 1) / (max_steps + 2), desc="Post Process")
        # self.run_tree_seg_2d_post() # TODO: have bug
        progress(1., desc="Completed")
        return self.update_2d_results()

    def run_tree_seg_2d_stage1(self):
        if self._image is None:
            print(f'[Web] run_tree_seg_2d_stage1 no image')
            return
        super().run_tree_seg_2d_stage1()
        print(f"[Web] complete run_tree_seg_2d_stage1")
        return self.update_2d_results()

    def run_tree_seg_2d_stage2(self):
        if self._image is None:
            print(f'[Web] run_tree_seg_2d_stage2 no image')
            return
        super().run_tree_seg_2d_stage2()
        print(f"[Web] complete run_tree_seg_2d_stage2")
        return self.update_2d_results()

    def run_tree_seg_2d_post(self):
        if self._tree_2d is None:
            print(f'[Web] run_tree_seg_2d_stage2 no tree2d')
            return
        super().run_tree_seg_2d_post()
        print(f"[Web] complete run_tree_seg_2d_post")
        return self.update_2d_results()

    def build_tree_seg_2d_ui(self):
        with gr.Row():
            image_select_btn = gr.Button('Image Gallery')
            image_upload_btn = gr.UploadButton('Upload Image', file_types=['image'], type='file', file_count='multiple')

        self.gallery = gr.Gallery(
            label='Images',
            show_label=False,
            elem_id='gallery',
            columns=[4],
            object_fit='contain',
            height='auto',
            visible=True)

        image_select_btn.click(self.load_images, None, self.gallery)
        image_upload_btn.upload(self.upload_images, image_upload_btn, self.gallery)
        self.image_box = gr.Image(tool='editor', source='upload', interactive=True, show_label=False)
        self.gallery.select(self.change_image, None, self.image_box)

        with gr.Row():
            reset_btn = gr.Button('Reset')
            stage1_btn = gr.Button('Grid Sample')
            stage2_btn = gr.Button('Heuristic Sample')
            post_btn = gr.Button('Post Process')
            autorun_btn = gr.Button('Auto Run')
        self.img_levels = [gr.Image(label=f"level {i}", interactive=False, visible=i == 0) for i in range(10)]

        reset_btn.click(self.reset_2d, outputs=self.img_levels)
        stage1_btn.click(self.run_tree_seg_2d_stage1, outputs=self.img_levels)
        stage2_btn.click(self.run_tree_seg_2d_stage2, outputs=self.img_levels)
        post_btn.click(self.run_tree_seg_2d_post, outputs=self.img_levels)
        autorun_btn.click(self.autorun_tree_seg_2d, outputs=self.img_levels)

    def change_mesh(self, *args, **kwargs):
        self.mesh_index = args[0]
        mesh_path = self.mesh_dir.joinpath(self.mesh_paths[self.mesh_index])
        self.mesh_path = mesh_path
        self._mesh = None
        print(f"[Web] change mesh to {self.mesh_path}")
        if self.mesh_path.suffix != '.glb':
            name = self.mesh_paths[self.mesh_index].replace('/', '_')
            mesh_path = self.cache_web.joinpath(name).with_suffix('.glb')
            if not mesh_path.exists():
                utils.save_glb(mesh_path, self.mesh)
                print(f"Web: convert to glb formart, save to {mesh_path}")
        assert mesh_path.is_file()
        return mesh_path

    def load_tree_seg_3d(self):
        if self._mesh is None:
            return
        self._tree_3d = Tree3Dv2(self.mesh, self.device)
        self.tree3d.load(Path('./results/Replica/room_1/n10000.tree3dv2'))
        self.levels_3d = self.tree3d.get_levels()
        restuls = []
        for i in range(len(self.seg3d_levels)):
            if i + 1 < len(self.levels_3d):
                mesh_path = self.mesh_path.with_name(f"{self.mesh_path.stem}_l{i + 1}.glb")
                if not mesh_path.exists():
                    mesh_l = self.mesh.clone()
                    seg_l = self.tree3d.masks[self.levels_3d[i + 1] - 1, 1:]
                    if mesh_l.v_clr is None:
                        mesh_l.v_clr = torch.full(mesh_l.v_pos.shape, 0, dtype=torch.float, device=self.device)
                    else:
                        mesh_l.v_clr = mesh_l.v_clr[..., :3]
                    colors = torch.from_numpy(np.array(sns.color_palette(n_colors=len(seg_l)))).to(mesh_l.v_clr)
                    face_colors = mesh_l.v_clr[mesh_l.f_pos].mean(dim=1)
                    for j in range(len(seg_l)):
                        face_colors[seg_l[j], :] = colors[j]
                    face_colors = face_colors[:, None, :].expand(face_colors.shape[0], 3, 3).reshape(-1, 3)
                    mesh_l.v_clr.index_reduce_(0, mesh_l.f_pos.view(-1).long(), face_colors, 'mean', include_self=False)
                    utils.save_glb(mesh_path, mesh_l)
                restuls.append(self.seg3d_levels[i].update(mesh_path, visible=True))
            else:
                restuls.append(self.seg3d_levels[i].update(value=None, visible=False))
        return restuls

    def build_tree_seg_3d_ui(self):
        self.mesh_paths = search_folder(self.mesh_dir, utils.mesh_extensions)
        n = len(self.mesh_dir.parts)
        self.mesh_paths = [Path(*path.parts[n:]).as_posix() for path in self.mesh_paths]
        self.change_mesh_ui = gr.Dropdown(
            choices=self.mesh_paths,
            value=self.mesh_paths[0] if len(self.mesh_paths) > 0 else None,
            type="index",
            multiselect=False,
            allow_custom_value=False,
            interactive=True,
            label='examples',
        )
        print(self.mesh_paths)
        self.view_3d = gr.Model3D(clear_color=(0., 0., 0., 0.))
        self.change_mesh_ui.select(self.change_mesh, self.change_mesh_ui, self.view_3d)
        with gr.Row():
            # self.load_3d = gr.Button('Load example')
            self.run_3d = gr.Button('Run')
        # self.load_3d.click(self.get_mesh_path, self.view_3d, self.view_3d)
        self.seg3d_levels = [gr.Model3D(label=f"level {i}", visible=False) for i in range(10)]
        self.run_3d.click(self.load_tree_seg_3d, outputs=self.seg3d_levels)


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.7, 0)
    WebUI(None).run()
