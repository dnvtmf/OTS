import os
from typing import Dict

import torch
import torch.hub
import torch.utils.model_zoo as model_zoo

# from extension import logger, Registry
from .utils import convert_pth


class BackboneBase(torch.nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self._stages = {}  # type:  Dict[int, torch.nn.Module]
        self._channels = {}  # type: Dict[int, int]
        self.download_url = ''
        self.pretrained_cfg = {}
        # if len(kwargs) > 0:
        #     ext.logger.WARN(f'Unused parametrs for {self.__class__.__name__}: {kwargs.keys()}')

    def forward(self, images) -> Dict[str, torch.Tensor]:
        x = images
        features = {'1': x}
        for stride, stage in self.stages.items():
            x = stage(x)
            features[str(stride)] = x
        return features

    @property
    def stages(self, by_stride=True):
        """return blocks split by downsample ratio"""
        return self._stages

    @property
    def channels(self):
        """return the channels for each blocks"""
        return self._channels

    def load_pretrained_model(self, model_path=None, strict=True):
        if model_path is None:
            if self.download_url == '':
                print('Can not load pretrained models from url: "{}"'.format(self.download_url))
                self.reset_parameters()
                return
            if isinstance(self.download_url, (list, tuple)):
                model_dir = os.path.join(torch.hub.get_dir(), 'checkpoints')
                model_path = os.path.join(model_dir, self.download_url[1])
                if not os.path.exists(model_path):
                    print(f'Please download pretrained-model from {self.download_url[0]}, '
                          f'then rename and put to {model_path}')
                    exit(1)
                pth = torch.load(model_path, map_location='cpu')
                model_path = self.download_url[0]
            else:
                pth = model_zoo.load_url(self.download_url, map_location='cpu')
                model_path = self.download_url
        else:
            if model_path == '':
                print("Do not load pretrained models")
                self.reset_parameters()
                return
            model_path = os.path.expanduser(model_path)
            if os.path.isfile(model_path):
                pth = torch.load(model_path, map_location='cpu')
            else:
                print(f'Can not load pretrained backbone from file "{model_path}"!!!')
                return

        pth = convert_pth(pth, **self.pretrained_cfg)
        missing_keys, unexpected_keys = self.load_state_dict(pth, strict=False)
        if len(missing_keys) > 0:
            if strict:
                raise RuntimeError(f"{missing_keys} is missed")
            else:
                print("\tmiss keys: {}".format(missing_keys))
        if len(unexpected_keys) > 0:
            print("\tunload keys: {}".format(unexpected_keys))
        print("==> Load pretrained model {} from {}".format(self.__class__.__name__, model_path))

    def reset_parameters(self, *args, **kwargs):
        pass

    def freeze_stages(self, low_stride=1, high_stride=1024, bn=True):
        """
        需要注意bn在根节点下的情况, 如ResNet.bn1
        """
        for stride, stage in self.stages.items():
            if not (low_stride <= stride <= high_stride):
                continue
            if bn:
                for m in stage.modules():
                    if hasattr(m, 'freeze'):
                        getattr(m, 'freeze')()
            for param in stage.parameters():
                param.requires_grad_(False)
        print(f'Freeze stages [{low_stride}-{min(high_stride, max(self.stages.keys()))}]{"and bn" if bn else ""}')

# BACKBONES = Registry()
