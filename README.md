# OTS

The offical implementaion of "Open-set Hierarchical Semantic Segmentation for 3D Scene".

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

## Citation

```text
@InProceedings{icme24_ots,
  title = 	 {Open-set Hierarchical Semantic Segmentation for 3D Scene},
  author =       {Wan, Diwen and Tang, Jiaxiang and Wang, Jingbo and Chen, Xiaokang and Gan, Lingyun and Zeng, Gang},
  booktitle = 	 {IEEE Conference on Multimedia Expo 2024},
  year = 	 {2024},
}

```