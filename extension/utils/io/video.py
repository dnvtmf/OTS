import numpy as np
from pathlib import Path
import imageio

__all__ = ['video_extensions', 'load_video', 'load_video_meta', 'save_video', 'save_gif', 'save_mp4']

video_extensions = ['.gif', '.mp4', '.mov', '.avi', '.mkv', '.flv', '.h264']


def load_video(filename: Path, **kwargs) -> np.ndarray:
    return imageio.v3.imread(filename, **kwargs)


def load_video_meta(filename: Path, **kwargs):
    return imageio.v3.imread(filename, **kwargs), imageio.v3.immeta(filename)


def save_video(filename: Path, images, fps=30, quality=8, **kwargs):
    if filename.suffix == '.mp4' or filename.suffix in '.gif':
        imageio.v3.imwrite(filename, images, fps=fps, quality=quality, **kwargs)
    else:
        raise NotImplementedError(f"Not supoort to save video as {filename.suffix} format!!")


def save_mp4(filename: Path, images, fps=30, **kwargs):
    save_video(filename.with_suffix('.mp4'), images, fps, **kwargs)


def save_gif(filename: Path, images, fps=30, loop=0, **kwargs):
    save_video(filename.with_suffix('.gif'), images, fps, loop=loop, **kwargs)


def test():
    from extension.utils import show_shape
    images, meta = load_video_meta(Path('~/Pictures/BvbFcCSsTv6Pd9BJpHvZfoK1W0KMraBi.gif'))
    print('\n', 'number of images:', len(images), type(images))
    print(show_shape(images))
    print(meta)
    save_video(Path('~/Pictures/test.gif'), images, loop=0)
