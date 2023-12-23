# TreeSeg

Tree Segmentation for 3D

## Install

```shell
conda env create -f enviroment.yaml
pip install git+https://github.com/NVlabs/nvdiffrast
conda activate tree_seg

cd semantic_sam/ops
bash ./make.sh

cd tree_segmentation/extension/_C
python setup.py build_ext --inplace
```

## Download pretrained weights

Download `sam_vit_h_4b8939.pth` from [Segment Anything](https://github.com/facebookresearch/segment-anything) in floder `weights`

Download `swinl_only_sam_many2many.pth` and `swint_only_sam_many2many.pth` from  [Sematic-SAM](https://github.com/UX-Decoder/Semantic-SAM/tree/main) in floder `weights`

## Acknowledgement

- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Sematic-SAM](https://github.com/UX-Decoder/Semantic-SAM/tree/main)

## TODO
- [ ] [3D] 只有一个mask的情况 
- [ ] [3D] 改正match loss
- [ ] [3D] Replica 选择视角
- [ ] [3D] Replica 使用高清图片
- [ ] [2D] metric set ignore nodes
- [ ] [2D] run experiment on PASCAL VOC/Cityscapes
- [ ] [3D] 基于边的融合方法
- [ ] [3D] add post-propocess
- [ ] [GUI] 固定渲染颜色
- [ ] [GUI] Edit 2D: 编辑模式
- [ ] [GUI] Edit 2D: 右键单击显示上一层和下一层
- [ ] [GUI] 合并save和load, 增加删除按钮
- [ ] [3D] Tree3Dv2._get_masks 使用CUDA实现
- [ ] [Paper] 补充材料
- [ ] [Paper] Demo视频
- [ ] [3D] 简化或细分Mesh
- [ ] [3D] 合并Tree3D和Tree3Dv2
- [ ] [2D] 使用RLE压缩Tree2D