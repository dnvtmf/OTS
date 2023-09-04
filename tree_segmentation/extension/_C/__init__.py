import warnings

_functions = {}


def get_python_function(name: str):
    return _functions[name]


try:
    from . import C_ext

    def try_use_C_extension(func):
        _functions[func.__name__] = func
        if hasattr(C_ext, func.__name__):
            return getattr(C_ext, func.__name__)
        else:
            warnings.warn(f'No such function in C/CPP/CUDA extension: {func.__name__}')
            return func

    def get_C_function(name: str):
        return getattr(C_ext, name, None)

    def have_C_functions(*names):
        for name in names:
            if not hasattr(C_ext, name):
                return False
        return True
except ImportError as e:
    warnings.warn(f'Please Compile C/CPP/CUDA code to use some functions. {e}')

    def try_use_C_extension(func):
        _functions[func.__name__] = func
        return func

    def get_C_function(name: str):
        warnings.warn(f'Please Compile C/CPP/CUDA code get function: {name}')
        return None

    def have_C_functions(*names):
        return False


__all__ = ['try_use_C_extension', 'get_C_function', 'get_python_function']
