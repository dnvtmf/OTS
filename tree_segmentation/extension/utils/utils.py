import time
from itertools import repeat
from typing import Any, Union, Tuple, Callable


def identity_fn(x, *args, **kwargs):
    """return first input"""
    return x


def Identity_fn(*args, **kwargs):
    """return an identity function"""
    return identity_fn


def extend_list(l: list, size: int, value=None):
    if value is None:
        value = l[-1]
    while len(l) < size:
        l.append(value)
    return l[:size]


def get_RAM_memory(unit=1024 ** 3):
    import psutil
    mem = psutil.virtual_memory()
    return [mem.total / unit, mem.used / unit, mem.cached / unit, mem.free / unit]
    ## Return RAM information (unit=kb) in a list
    ## Index 0: total RAM
    ## Index 1: used RAM
    ## Index 2: free RAM
    # p = os.popen('free -m')
    # i = 0
    # while 1:
    #     i = i + 1
    #     line = p.readline()
    #     if i == 2:
    #         return line.split()[1:4]


class Config:

    def __init__(self, **entries):
        for k, v in entries.items():
            setattr(self, k, v)

    def __repr__(self):
        return f'Config({", ".join(f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_"))})'


def n_tuple(x, n: int) -> tuple:
    if isinstance(x, (tuple, list, set)):
        assert len(x) == n, f"The length is {len(x)} not {n}"
        return tuple(x)
    return tuple(repeat(x, n))


def to_list(x) -> list:
    return list(x) if isinstance(x, (tuple, list, set)) else [x]


def make_divisible(x: Union[int, float], size_divisible: Union[int, Tuple[int, int]]):
    """令y=size_divisible[0] * k + size_divisible[1], 且 y >= x"""
    x = int(x)
    if isinstance(size_divisible, int):
        return ((x - 1) // size_divisible + 1) * size_divisible
    else:
        return ((x - 1 - size_divisible[1]) // size_divisible[0] + 1) * size_divisible[0] + size_divisible[1]


def get_colors(num_colors: int, to_255=True):
    import colorsys
    import random

    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        lightness = (50 + random.uniform(-1, 1) * 10) / 100.
        saturation = (90 + random.uniform(-1, 1) * 10) / 100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    if to_255:
        colors = [tuple([int(c * 255) for c in rgb]) for rgb in colors]
    return colors


def first_not_none(*args):
    for x in args:
        if x is not None:
            return x
    return None


fnn = first_not_none


# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(args):
        if isinstance(args, list):
            return [wrapper(x) for x in args]
        elif isinstance(args, tuple):
            return tuple([wrapper(x) for x in args])
        elif isinstance(args, dict):
            return {k: wrapper(v) for k, v in args.items()}
        else:
            return func(args)

    return wrapper


def check_interval(step: int, interval: int, total: int = None):
    return interval > 0 and (step % interval == 0 or step == total)


def change_each(data: Any, func: Callable):
    """change each item for list, tuple and dict.values()"""
    if isinstance(data, list):
        data = [change_each(x, func) for x in data]
    elif isinstance(data, tuple):
        data = tuple(change_each(x, func) for x in data)
    elif isinstance(data, dict):
        data = {k: change_each(v, func) for k, v in data.items()}
    return func(data)


def test_change_each():
    func = lambda x: x
    print()
    print(change_each(123, func))
    print(change_each([123, '456'], func))
    print(change_each([123, '456', {7: 8}], func))
    print(change_each([123, '456', {7: 8}, (9,)], func))
    print(change_each((9,), func))
    print(change_each({1: [2, 3], 2: (3,), 4: {5: 6}}, func))


def get_date() -> str:
    return time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))


if __name__ == '__main__':
    # test_eval_str()
    # for n in range(-20, 20):
    #     print(float2str(1.23456789012345 * 10 ** n, 6, precision=3))
    #     print(float2str(1.23 * 10 ** n, 6))
    test_change_each()
