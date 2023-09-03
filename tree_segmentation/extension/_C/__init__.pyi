from typing import List, Callable, Optional

from torch import Tensor


def try_use_C_extension(func: Callable) -> Callable:
    ...


def get_C_function(name: str) -> Optional[Callable]:
    ...


def get_python_function(name: str) -> Callable:
    ...


def have_C_functions(*names: str) -> bool:
    ...
