import errno
import os

import torch
import torch.distributed as dist


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def patchify(imgs, patch_size):
    h = w = imgs.shape[2] // patch_size
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, patch_size, w, patch_size))
    x = torch.einsum("nchpwq->nhwpqc", x)
    x = x.reshape(shape=(imgs.shape[0], h * w, patch_size**2 * 3))
    return x


def unpatchify(x, patch_size):
    h = w = int(x.shape[1] ** 0.5)
    x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, 3))
    x = torch.einsum("nhwpqc->nchpwq", x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * patch_size, h * patch_size))
    return imgs
