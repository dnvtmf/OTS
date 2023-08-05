# TreeSeg

Tree Segmentation for 3D

## Overleaf

https://www.overleaf.com/6945515643cnsqpwznzgsq

## Install

```shell
conda env create -f enviroment.yaml
pip install git+https://github.com/NVlabs/nvdiffrast
conda activate tree_seg

cd semantic_sam/ops
bash ./build.sh
```

## Download pretrained weights

Download `sam_vit_h_4b8939.pth` from [Segment Anything](https://github.com/facebookresearch/segment-anything) in floder `weights`

Download `swinl_only_sam_many2many.pth` and `swint_only_sam_many2many.pth` from  [Sematic-SAM](https://github.com/UX-Decoder/Semantic-SAM/tree/main) in floder `weights`

## Acknowledgement

- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Sematic-SAM](https://github.com/UX-Decoder/Semantic-SAM/tree/main)

## TODO

- [x] [2D] add post propocess
- [x] [3D] Tree3Dv2 to tree
- [x] [3D] 添加tree loss

### 8.3 work list 
- [x] [2D] 修改metric, 在SA-1B上测试metric结果
- [x] [3D] 修改3D loss, 增加有关G重建的loss

### 8.4 work list
- [x] [Paper] 撰写论文的方法部分

### 8.5 work list
- [ ] [2D] 完成2D的实验
- [ ] [2D] 基于SA-1B选取超参数
- [ ] [3D] 快速2D分割
- [ ] [Paper] 撰写实验部分

### future
- [ ] [2D] 测试automatic mask generator的结果
- [ ] [2D] metric set ignore nodes
- [ ] [2D] run experiment on PASCAL VOC/Cityscapes
- [ ] [3D] 基于边的融合方法
- [ ] [3D] add post-propocess
- [ ] [3D] PartNet使用纹理
- [ ] [GUI] 固定渲染颜色
- [ ] [GUI] Edit 2D: 编辑模式
- [ ] [GUI] Edit 2D: 右键单击显示上一层和下一层
- [ ] [GUI] 合并save和load, 增加删除按钮
- [ ] [3D] Tree3Dv2._get_masks 使用CUDA实现
- [ ] [Paper] 撰写相关工作
- [ ] [Paper] 撰写Introduction
- [ ] [Paper] 补充材料
- [ ] [Paper] Demo视频
- [ ] [3D] 简化或细分Mesh
- [ ] [3D] 合并Tree3D和Tree3Dv2
