import logging
import re
import math
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from functools import wraps
from typing import Union, Tuple
from itertools import repeat

import torch
from torch import Tensor


@contextmanager
def _ignore_torch_cuda_oom():
    """
    A context which ignores CUDA OOM exception from pytorch.
    """
    try:
        yield
    except RuntimeError as e:
        # NOTE: the string may change?
        if "CUDA out of memory. " in str(e):
            pass
        else:
            raise


def retry_if_cuda_oom(func):
    """
    Makes a function retry itself after encountering
    pytorch's CUDA OOM error.
    It will first retry after calling `torch.cuda.empty_cache()`.

    If that still fails, it will then retry by trying to convert inputs to CPUs.
    In this case, it expects the function to dispatch to CPU implementation.
    The return values may become CPU tensors as well and it's user's
    responsibility to convert it back to CUDA tensor if needed.

    Args:
        func: a stateless callable that takes tensor-like objects as arguments

    Returns:
        a callable which retries `func` if OOM is encountered.

    Examples:
    ::
        output = retry_if_cuda_oom(some_torch_function)(input1, input2)
        # output may be on CPU even if inputs are on GPU

    Note:
        1. When converting inputs to CPU, it will only look at each argument and check
           if it has `.device` and `.to` for conversion. Nested structures of tensors
           are not supported.

        2. Since the function might be called more than once, it has to be
           stateless.
    """

    def maybe_to_cpu(x):
        try:
            like_gpu_tensor = x.device.type == "cuda" and hasattr(x, "to")
        except AttributeError:
            like_gpu_tensor = False
        if like_gpu_tensor:
            return x.to(device="cpu")
        else:
            return x

    @wraps(func)
    def wrapped(*args, **kwargs):
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Clear cache and retry
        torch.cuda.empty_cache()
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Try on CPU. This slows down the code significantly, therefore print a notice.
        logger = logging.getLogger(__name__)
        logger.info("Attempting to copy inputs of {} to CPU due to CUDA OOM".format(str(func)[0:5]))
        new_args = (maybe_to_cpu(x) for x in args)
        new_kwargs = {k: maybe_to_cpu(v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)

    return wrapped


def state_dict_strip_prefix_if_present(state_dict, prefix='module.') -> OrderedDict:
    if not all(key.startswith(prefix) for key in state_dict.keys()):
        return state_dict
    n = len(prefix)
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key[n:]] = value
    return stripped_state_dict


def state_dict_add_prefix_if_not_present(state_dict, prefix) -> OrderedDict:
    if any(key.startswith(prefix) for key in state_dict.keys()):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[prefix + key] = value
    return stripped_state_dict


def convert_pth(pth: dict, key=None, replace: dict = None, bgr_to_rgb=False, first_layer_name=None):
    # get pretrained model in checkpoint
    not_a_checkpoint = not (isinstance(pth, dict) and all(isinstance(v, Tensor) for v in pth.values()))
    if not_a_checkpoint:
        if key is None or key not in pth:  # gauss correct key
            key = None
            for gauss_key in ['state_dict', 'model', 'network']:
                if gauss_key in pth:
                    key = gauss_key
        if key is not None:
            pth = pth[key]
            not_a_checkpoint = not (isinstance(pth, dict) and all(isinstance(v, Tensor) for v in pth.values()))
    if not_a_checkpoint:
        print(f'The loaded pth is a checkpoint, please point out the key of state_dict in {pth.keys()}')
        return {}

    # deal first layer
    if bgr_to_rgb:
        if first_layer_name is None:
            first_layer_name = list(pth.keys())[0]
        v = pth[first_layer_name]
        assert v.ndim == 4 and v.size(1) == 3, f"{first_layer_name} is not first layer"
        pth[first_layer_name] = pth[first_layer_name][:, (2, 1, 0), :, :]
    # if Image.CHANNELS > 3:
    #     for name, value in pth.items():  # type: str, torch.Tensor
    #         if value.ndim == 4 and value.size(1) == 3:
    #             additional_shape = value.shape[0], Image.CHANNELS - 3, *value.shape[2:]
    #             pth[name] = torch.cat([value, value.new_zeros(additional_shape)], dim=1)

    pth = state_dict_strip_prefix_if_present(pth, 'module.')
    # 基于正则表达，重命名pth中的key
    if replace is not None:
        new_pth = OrderedDict()
        for k, v in pth.items():
            deal_time = 0
            for a, b in replace.items():
                match_result = re.match(a, k)
                if match_result:
                    k = b.format(*match_result.groups())
                    if k:  # skip this weight
                        new_pth[k] = v
                    deal_time += 1
                    break  # 多个匹配时，在前的先起作用
            if deal_time == 0:  # 不在替换列表中，保持原样
                new_pth[k] = v
            # assert deal_time == 1, f"Please check replace list, error for {k}"
    else:
        new_pth = pth
    return new_pth


def make_divisible(x: Union[int, float], size_divisible: Union[int, Tuple[int, int]]):
    """令y=size_divisible[0] * k + size_divisible[1], 且 y >= x"""
    x = int(x)
    if isinstance(size_divisible, int):
        return ((x - 1) // size_divisible + 1) * size_divisible
    else:
        return ((x - 1 - size_divisible[1]) // size_divisible[0] + 1) * size_divisible[0] + size_divisible[1]


def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
            stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


def n_tuple(x, n: int) -> tuple:
    if isinstance(x, (tuple, list, set)):
        assert len(x) == n, f"The length is {len(x)} not {n}"
        return tuple(x)
    return tuple(repeat(x, n))
