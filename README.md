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
- [x] [2D] 基于SA-1B选取超参数
- [ ] [3D] 快速2D分割
- [ ] [Paper] 撰写实验部分
### 8.6 work list
- [x] [Paper] Introduction草稿
- [x] [3D] PartNet选用实验划分
- [x] [3D] 新的建图方式

### 8.7 work list
- [x] [2D] 完成2D对比实验 origin
- [x] [3D] 完成loss改进
- [ ] [3D] PartNet gt_seg

### 8.8 work list
- [x] [3D] PartNet gt_seg
- [ ] [2D] 完成2D对比实验 origin_tree
- [x] [2D] 完成2D对比实验 0.5
- [ ] [3D] indoor dataset
- [x] [3D] outdoor dataset
- [x] [3D] 室外场景
- [ ] [Paper] 撰写相关工作
- [x] [3D] PartNet使用纹理

### 8.9  
- [x] [Paper] 撰写Introduction
- [x] [2D] 完成2D对比实验 0
- [x] [2D] 完成2D对比实验 1.0

### 8.10 work list
- [ ] [3D] 快速2D分割 (failed)
- [x] [3D] view选择
- [x] [Paper] 插图绘制
- [ ] [3D] 测试样本筛选

### 8.11 work list
- [ ] [3D] 不同类型GNN测试
- [ ] [3D] 不同loss functions测试
- [ ] [3D] 不同K测试
- [x] [Paper] 完成PartNet的插图绘制
- [ ] [Paper] 完成场景的插图绘制
- [ ] [Paper] 完成实验部分撰写

### future
- [ ] [2D] 测试automatic mask generator的结果
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