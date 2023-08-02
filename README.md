# TreeSeg

Tree Segmentation for 3D

## Overleaf

https://www.overleaf.com/6945515643cnsqpwznzgsq

## Install

```shell
conda env create -f enviroment.yaml
pip install git+https://github.com/NVlabs/nvdiffrast
conda activate tree_seg
```

## Download pretrained weights

Download `sam_vit_h_4b8939.pth` from [Segment Anything](https://github.com/facebookresearch/segment-anything) in floder `weights`

Download `swinl_only_sam_many2many.pth` and `swint_only_sam_many2many.pth` from  [Sematic-SAM](https://github.com/UX-Decoder/Semantic-SAM/tree/main) in floder `weights`

## Acknowledgement

- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Sematic-SAM](https://github.com/UX-Decoder/Semantic-SAM/tree/main)

## TODO

- [x] [2D] add post propocess
- [ ] [2D] metric set ignore nodes
- [ ] [2D] run experiment on PASCAL VOC/Cityscapes
- [x] [3D] Tree3Dv2 to tree
- [ ] [3D] 添加tree loss
- [ ] [3D] 快速2D分割
- [ ] [3D] 基于边的融合方法
- [ ] [3D] add post-propocess
- [ ] [GUI] 固定渲染颜色
- [ ] [GUI] Edit 2D: 编辑模式
- [ ] [GUI] Edit 2D: 右键单击显示上一层和下一层
- [ ] [GUI] 合并save和load, 增加删除按钮
- [ ] [3D] Tree3Dv2._get_masks 使用CUDA实现
- [ ] [paper] start write