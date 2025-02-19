import os
from pathlib import Path
import time
from typing import Union, Any

import cv2
from PIL import Image
import gradio as gr
import numpy as np
import torch_geometric as pyg
import seaborn as sns
import torch
import math
from tqdm import tqdm

from segment_anything.build_sam import Sam
from semantic_sam import SemanticSAM
from tree_segmentation import Tree3D
from tree_segmentation.extension import utils, ops_3d
from tree_segmentation.tree_segment import TreeSegment
from tree_segmentation.util import get_colored_masks, search_folder, get_hash_name


# 随机视角
def random_camera_pose(
    image_size=1024,
    num=1,
    radius_range=(2., 4.),
    elev_range=(0, 180),
    azim_range=(-180, 180),
    device=torch.device('cuda')
):
    radius = torch.rand(num, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]
    thetas = torch.rand(num, device=device) * (elev_range[1] - elev_range[0]) + elev_range[0]
    phis = torch.rand(num, device=device) * (azim_range[1] - azim_range[0]) + azim_range[0]

    fovy = math.radians(60)
    Tv2c = ops_3d.perspective(fovy=fovy, size=(image_size, image_size), device=device)
    eye = ops_3d.coord_spherical_to(radius, thetas.deg2rad(), phis.deg2rad()).to(device)
    Tw2v = ops_3d.look_at(eye, torch.zeros_like(eye))
    Tv2w = ops_3d.look_at(eye, torch.zeros_like(eye), inv=True)
    Tw2c = Tv2c @ Tw2v
    return Tw2c, Tv2w


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

    def get_value(self, name, default=None) -> Any:
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
        with gr.Group():
            gr.Markdown('Tree Segmentation 3D Options')
            with gr.Row():
                add_option('num_images', 100, precision=0)

    def update_2d_results(self):
        data = []
        for i in range(len(self.img_levels)):
            if i + 1 < len(self.levels_2d):
                assert 0 < self.levels_2d[i + 1].min() and self.levels_2d[i + 1].max() < len(self.tree2d.parent)
                masks = self.tree2d.masks[self.levels_2d[i + 1] - 1]
                image = get_colored_masks(masks)
                alpha = self.get_value('alpha', 0.3)
                if self.image.dtype == np.uint8:
                    image = (image * 255).astype(np.uint8)
                image = cv2.addWeighted(self.image[..., :3], alpha, image[..., :3], 1 - alpha, 0)
                data.append(self.img_levels[i].update(image, visible=True))
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
            mesh_name = self.mesh_paths[self.mesh_index].replace('/', '_')
            mesh_path = self.cache_web.joinpath(mesh_name).with_suffix('.glb')
            if not mesh_path.exists():
                utils.save_glb(mesh_path, self.mesh)
                print(f"Web: convert to glb formart, save to {mesh_path}")
        assert mesh_path.is_file()
        self.load_mesh(obj_path=self.mesh_path)  # 加载mesh
        self.tree_3d = Tree3D(self.mesh, self.device, verbose=1)
        return mesh_path

    def load_tree_seg_3d(self):
        self.cache_2d_all = self.cache_web.joinpath("2d_all")
        if self._mesh is None:
            print("[Web] self._mesh is None")
            return
            # self._tree_3d = Tree3Dv2(self.mesh, self.device)
        # self.tree3d.load(Path('./results/Replica/room_1/n10000.tree3dv2'))
        # self.tree3d.load(Path('./results/gt.tree3dv2'))
        self.cache_tree3d = self.cache_web.joinpath(f'my_{self.mesh_name}.tree3dv2')
        if not self.cache_tree3d.exists():
            print("[Web] Tree 3D not exists, loading Tree 2D data.")
            self.tree_3d = Tree3D(self.mesh, self.device, verbose=1)
            self.tree_3d.load_2d_results(self.cache_2d_all, pack=True)  # 读取.data文件
            print("[Web] Tree 3D building graph...")
            A = self.tree_3d.build_all_graph()
            X, autoencoder = self.tree_3d.compress_masks(epochs=3000)
            K = self.tree_3d.Lmax * 2
            gnn = pyg.nn.GCN(in_channels=X.shape[1],
                hidden_channels=128,
                num_layers=2,
                out_channels=K,
                norm='BatchNorm').cuda()
            print(gnn)
            print("[Web] Tree 3D running...")
            self.tree_3d.run(epochs=10000, K=K, gnn=gnn, A=A * A.ge(0.5), X=X)
            self.tree_3d.save(self.cache_tree3d)
            print("[Web] Tree 3D done.")
        else:
            print("[Web] Tree 3D already exists, loading from cache.")
            self.tree_3d.load(self.cache_tree3d)

        self.levels_3d = self.tree_3d.get_levels()
        restuls = []
        for i in range(len(self.seg3d_levels)):
            if i + 1 < len(self.levels_3d):
                mesh_path = self.mesh_path.with_name(f"{self.mesh_path.stem}_l{i + 1}.glb")
                if not mesh_path.exists():
                    mesh_l = self.mesh.clone()
                    seg_l = self.tree_3d.masks[self.levels_3d[i + 1] - 1, 1:]
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
        print("[Web] load_tree_seg_3d done.")
        # pprint(restuls[0]) # 调试用，发现数据为空
        return restuls

    # 生成图片
    def render(self):
        n = self.get_value('num_images', 100)
        # n为图片数量
        self.mesh_name = self.mesh_paths[self.mesh_index].replace('/', '_').split(".")[0]
        print(self.mesh_name)
        self.cache_render = self.cache_web.joinpath("render_cache")
        self.cache_render.mkdir(exist_ok=True)
        self.cache_render = self.cache_render.joinpath(self.mesh_name)
        self.cache_render.mkdir(exist_ok=True)

        self.image_list = []
        self.tri_ids = []
        self.Tw2vs = []
        for i in tqdm(range(n)):
            i = str(i)
            img_path = self.cache_render.joinpath(f"pic_{i}").with_suffix('.png')
            tri_id_path = self.cache_render.joinpath(f"tri_id_{i}").with_suffix(".pth")
            Tw2v_path = self.cache_render.joinpath(f"Tw2v_{i}").with_suffix(".pth")
            if not img_path.exists() and not tri_id_path.exists() and not Tw2v_path.exists():
                print(f"[Web] Rendering picture {i}, save to {img_path}.")
                _, Tv2w = random_camera_pose()
                # self.new_camera_pose(Tw2v=Tv2w[0].inverse())
                Tw2v = Tv2w[0].inverse()
                img, tri_id = self.rendering(Tw2v=Tw2v)
                img = (np.array(img) * 255).astype(np.uint8)  # 这里为保存为png将其乘255，后续分割时需做逆变换，否则分割无效
                # print(img.shape)
                Image.fromarray(img, 'RGB').save(img_path)
                torch.save(tri_id.clone(), tri_id_path)
                torch.save(Tw2v.clone(), Tw2v_path)
                # 问题：保存时需要转换成uint8，但转换会导致一些损失，是否需要考虑？
            else:
                print(f"[Web] {img_path} already exists, passing.")
                img = np.array(Image.open(img_path))
                tri_id = torch.load(tri_id_path)
                Tw2v = torch.load(Tw2v_path)
            self.image_list.append(img)
            self.tri_ids.append(tri_id)
            self.Tw2vs.append(Tw2v)
        print("[Web] Render finished.")
        return self.image_list

    # 只保存Tw2vs
    def render_v2(self):
        n = self.get_value('num_images', 100)
        # n为图片数量
        self.mesh_name = self.mesh_paths[self.mesh_index].replace('/', '_').split(".")[0]
        print(self.mesh_name)
        self.cache_Tw2vs = self.cache_web.joinpath(f"{self.mesh_name}_Tw2vs_{n}").with_suffix(".pth")
        self.Tw2vs = []
        self.image_list = []
        self.tri_ids = []
        if self.cache_Tw2vs.exists():
            print(f"[Web] {self.cache_Tw2vs} already exists, passing.")
            self.Tw2vs = torch.load(self.cache_Tw2vs)
        else:
            print(f"[Web] Rendering, save to {self.cache_Tw2vs}.")
            for i in range(n):
                _, Tv2w = random_camera_pose()
                Tw2v = Tv2w[0].inverse()
                self.Tw2vs.append(Tw2v.clone())
            torch.save(self.Tw2vs, self.cache_Tw2vs)
        for i in tqdm(range(n)):
            img, tri_id = self.rendering(Tw2v=self.Tw2vs[i])
            # img = (np.array(img)*255).astype(np.uint8) # 不保存为png，则无需转化
            self.image_list.append(img)
            self.tri_ids.append(tri_id)
        print("[Web] Render finished.")
        return self.image_list

    def seg_2d_all(self):
        self.seg_all_list = []
        print(len(self.image_list))
        for index in tqdm(range(len(self.image_list))):
            # for image in tqdm(self.image_list):
            # 调试：判断是否为不同图片
            if self._image is None:
                print("[Web] New image")
            else:
                if (self._image == self.image_list[index]).all():
                    print("[Web] Old Image.")
                else:
                    print("[Web] New image")
                    self.reset_2d()
            self.image = self.image_list[index]
            self.tri_id = self.tri_ids[index]
            # 直接使用前面的2d代码
            # self.seg_all_list += [i["value"] for i in self.autorun_tree_seg_2d() if i["value"] is not None]

            # 使用父级的autorun
            super().run_tree_seg_2d_stage1()
            for i in range(len(self.levels_2d) - 1):
                masks = self.tree2d.masks[self.levels_2d[i + 1] - 1]
                image = get_colored_masks(masks)
                alpha = self.get_value('alpha', 0.3)
                if self.image.dtype == np.uint8:
                    image = (image * 255).astype(np.uint8)
                image = cv2.addWeighted(self.image[..., :3], alpha, image[..., :3], 1 - alpha, 0)
                # 本步骤有问题
                self.seg_all_list.append(image)

        return self.seg_all_list

    def seg_2d_all_v2(self):
        # 暂时不能显示结果，但保存的.data文件可以使Tree 3D运行，可正常进行分割并显示分割结果。
        self.cache_2d_all = self.cache_web.joinpath("2d_all")
        self.cache_2d_all.mkdir(exist_ok=True)
        self.seg_all_list = []

        # predictor = TreePredictor(self._model_type, stability_score_thresh=0.92, points_per_batch=256)
        for index in tqdm(range(len(self.image_list))):
            self.image = self.image_list[index]
            self.tri_id = self.tri_ids[index]
            tree_data = self.predictor.tree_generate(
                (self._image * 255).astype(np.uint8),  # 此处需要0-255图片
                max_steps=100,
                in_threshold=0.8,
                union_threshold=0.1,
                min_mask_region_area=100,
                points_per_update=256,
                device=self.device,
                in_thre_area=50,
            )
            data = {
                'tree_data': tree_data.save(filename=None),
                'tri_id': self.tri_id.clone(),
                'image': self._image,
                'Tw2v': self.Tw2vs[index].clone(),
            }
            torch.save(data, self.cache_2d_all.joinpath(f"view_{index:04d}.data"))
        print("[Web] 2D Seg all done.")
        return self.seg_all_list

    def build_tree_seg_3d_ui(self):
        self.mesh_paths = search_folder(self.mesh_dir, utils.mesh_extensions)
        n = len(self.mesh_dir.parts)
        self.mesh_paths = [Path(*path.parts[n:]).as_posix() for path in self.mesh_paths]
        self.change_mesh_ui = gr.Dropdown(
            choices=self.mesh_paths,
            value=None,
            type="index",
            multiselect=False,
            allow_custom_value=False,
            interactive=True,
            label='examples',
        )
        print(self.mesh_paths)
        self.view_3d = gr.Model3D(clear_color=[0., 0., 0., 0.])
        self.change_mesh_ui.select(self.change_mesh, self.change_mesh_ui, self.view_3d)
        with gr.Row():
            self.reset_3d_btn = gr.Button('Reset')
            self.render_btn = gr.Button('Render')
            self.run_3d_seg_2d = gr.Button('2D Seg')
            self.run_3d_merge = gr.Button('3D Seg')
        # with gr.Row():
        #     # self.show_2d = gr.Button('Show 2D')
        #     self.slider = gr.Slider(minimum=1, maximum=self.get_value('num_images', 100), interactive=True)
        with gr.Row():
            self.tree_2d_gallery = gr.Gallery(
                label='Images',
                show_label=False,
                elem_id='gallery',
                columns=[6],
                object_fit='contain',
                height='auto',
                visible=True)

        # self.load_3d.click(self.get_mesh_path, self.view_3d, self.view_3d)
        self.seg3d_levels = [gr.Model3D(label=f"level {i}", visible=True) for i in range(10)]
        self.run_3d_merge.click(self.load_tree_seg_3d, outputs=self.seg3d_levels)
        # self.reset_3d_btn.click(self.reset_3d, outputs=[self.slider])
        self.reset_3d_btn.click(self.reset_3d)
        self.render_btn.click(self.render_v2, outputs=self.tree_2d_gallery)  # 点击render, 生成n张图片
        # self.slider.release()               # 滑动滑块展示图片
        self.run_3d_seg_2d.click(self.seg_2d_all_v2, outputs=self.tree_2d_gallery)  # 点2D Seg按钮
        # self.tree_3d = Tree3Dv2(self.mesh, self.device)
        # self.run_3d_merge.click(self.tree_3d.run, outputs=self.seg3d_levels)             # 点击3D Seg按钮
        # self.show_2d.click(lambda: self.tree_2d_gallery(visible=not self.tree_2d_gallery.get_config()[
        #     'visible']),
        #     outputs=self.tree_2d_gallery)


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.7, 0)
    WebUI(None).run()
