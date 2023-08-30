import os
from typing import Optional, Union
import gradio as gr
import random
import numpy as np
import time
import shutil
from pathlib import Path
from segment_anything.build_sam import Sam
from semantic_sam import SemanticSAM

from tree_segmentation.extension import utils
from tree_segmentation.tree_3d import TreeSegment


class WebUI(TreeSegment):

    def __init__(self, args, model: Union[Sam, SemanticSAM, None] = None):
        super().__init__(args, model)
        self.image_dir = Path('./images').expanduser()
        self.image_index = 0
        self.image_paths = []

        with gr.Blocks() as tree_seg_2d_blocks:
            self.build_tree_seg_2d_ui()

        b = gr.Button('B')
        self.web_ui = gr.TabbedInterface([tree_seg_2d_blocks, b], ["2D Tree Segmentation", "3D Tree Segmentation"])

    def run(self):
        self.web_ui.launch(share=False, server_name='0.0.0.0')

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
        self._image = utils.load_image(self.image_paths[self.image_index])
        return self._image

    def build_tree_seg_2d_ui(self):

        with gr.Row():
            image_select_btn = gr.Button('Select Image')
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
        self.image_box = gr.Image(tool='sketch', source='upload', interactive=True, show_label=False)

        self.gallery.select(self.change_image, None, self.image_box)
        # section_labels = [
        #     "apple",
        #     "banana",
        #     "carrot",
        #     "donut",
        #     "eggplant",
        #     "fish",
        #     "grapes",
        #     "hamburger",
        #     "ice cream",
        #     "juice",
        # ]

        # with gr.Row():
        #     num_boxes = gr.Slider(0, 5, 2, step=1, label="Number of boxes")
        #     num_segments = gr.Slider(0, 5, 1, step=1, label="Number of segments")

        # with gr.Row():
        #     img_input = gr.Image()
        #     img_output = gr.AnnotatedImage(color_map={"banana": "#a89a00", "carrot": "#ffae00"})

        # section_btn = gr.Button("Identify Sections")
        # selected_section = gr.Textbox(label="Selected Section")

        # def section(img, num_boxes, num_segments):
        #     sections = []
        #     for a in range(num_boxes):
        #         x = random.randint(0, img.shape[1])
        #         y = random.randint(0, img.shape[0])
        #         w = random.randint(0, img.shape[1] - x)
        #         h = random.randint(0, img.shape[0] - y)
        #         sections.append(((x, y, x + w, y + h), section_labels[a]))
        #     for b in range(num_segments):
        #         x = random.randint(0, img.shape[1])
        #         y = random.randint(0, img.shape[0])
        #         r = random.randint(0, min(x, y, img.shape[1] - x, img.shape[0] - y))
        #         mask = np.zeros(img.shape[:2])
        #         for i in range(img.shape[0]):
        #             for j in range(img.shape[1]):
        #                 dist_square = (i - y)**2 + (j - x)**2
        #                 if dist_square < r**2:
        #                     mask[i, j] = round((r**2 - dist_square) / r**2 * 4) / 4
        #         sections.append((mask, section_labels[b + num_boxes]))
        #     return (img, sections)

        # section_btn.click(section, [img_input, num_boxes, num_segments], img_output)

        # def select_section(evt: gr.SelectData):
        #     return section_labels[evt.index]

        # img_output.select(select_section, None, selected_section)


if __name__ == "__main__":
    WebUI(None).run()
