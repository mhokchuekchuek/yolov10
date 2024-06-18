import numpy as np
from ultralytics.data.augment import LetterBox
import torch

def pre_transform(im):
    """
    Pre-transform input image before inference.

    Args:
        im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

    Returns:
        (list): A list of transformed images.
    """
    same_shapes = len({x.shape for x in im}) == 1
    # you should check img size from check_imgsz first
    letterbox = LetterBox(
        [640, 640], auto=same_shapes, stride=32
    )  # stride can be check from model.stride
    return [letterbox(image=x) for x in im]


def preprocess(im):
    """
    Prepares input image before inference.

    Args:
        im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
    """
    not_tensor = not isinstance(im, torch.Tensor)
    if not_tensor:
        im = np.stack(pre_transform(im))
        im = im[..., ::-1].transpose(
            (0, 3, 1, 2)
        )  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)

    im = im.to("cpu")
    im = im.float()  # uint8 to fp16/32
    if not_tensor:
        im /= 255  # 0 - 255 to 0.0 - 1.0
    return im
