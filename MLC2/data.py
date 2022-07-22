import io
import os
import random
from PIL import Image
from utils.parallelzipfile import ParallelZipFile as ZipFile

import torch
from transformers import ViTFeatureExtractor


def random_crop(image):
    w, h = image.size
    if w > h:
        crop_ratio_w = random.uniform(0.6, 1)
        crop_w = w * crop_ratio_w
        crop_ratio_h = random.uniform(crop_ratio_w, 1)
        crop_h = h * crop_ratio_h
    else:
        crop_ratio_h = random.uniform(0.6, 1)
        crop_h = h * crop_ratio_h
        crop_ratio_w = random.uniform(crop_ratio_h, 1)
        crop_w = w * crop_ratio_w

    x1 = random.randrange(0, max(int(w - crop_w), 1))
    y1 = random.randrange(0, max(int(h - crop_h), 1))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    image = image.crop((x1, y1, x2, y2))
    return image


def random_flip(image):
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        database = []
        for path, _, files in os.walk(cfg.data_dir):
            for name in files:
                database.append(os.path.join(path, name))
        self.database = database

        self.transform = ViTFeatureExtractor(
            do_resize=True,
            size=(cfg.image_size, cfg.image_size),
            do_center_crop=False,
            do_normalize=True
        )
        self.cfg = cfg

    def __getitem__(self, index):
        image = Image.open(self.database[index])
        image = image.convert("RGB")
        image = random_crop(image)
        image = random_flip(image)
        pixel_values = self.transform(image, return_tensors="pt").pixel_values

        num_patches = (self.cfg.image_size // self.cfg.patch_size) ** 2
        num_masked_patches = int(num_patches * self.cfg.patch_mask_ratio)
        _, masked_indices = torch.rand(num_patches).topk(num_masked_patches)
        patch_mask = torch.zeros(num_patches, dtype=torch.long)
        patch_mask[masked_indices] = 1

        return pixel_values, patch_mask.bool()

    def __len__(self):
        return len(self.database)


class ZipDatasetUnsafe(torch.utils.data.Dataset):
    def __init__(self, cfg):
        zipdata = ZipFile(cfg.zipfile, "r")
        database = zipdata.namelist()
        database = [item for item in database if ".jpg" in item]
        self.zipdata = zipdata
        self.database = database

        self.transform = ViTFeatureExtractor(
            do_resize=True,
            size=(cfg.image_size, cfg.image_size),
            do_center_crop=False,
            do_normalize=True
        )
        self.cfg = cfg

    def __getitem__(self, index):
        filename = self.database[index]
        image_raw = self.zipdata.read(filename)

        image = Image.open(io.BytesIO(image_raw))
        image = image.convert("RGB")
        image = random_crop(image)
        image = random_flip(image)
        pixel_values = self.transform(image, return_tensors="pt").pixel_values

        num_patches = (self.cfg.image_size // self.cfg.patch_size) ** 2
        num_masked_patches = int(num_patches * self.cfg.patch_mask_ratio)
        _, masked_indices = torch.rand(num_patches).topk(num_masked_patches)
        patch_mask = torch.zeros(num_patches, dtype=torch.long)
        patch_mask[masked_indices] = 1

        return pixel_values, patch_mask.bool()

    def __len__(self):
        return len(self.database)


def collate_fn(batch):
    batch = list(zip(*batch))
    batch_pixels_values = torch.cat(batch[0], dim=0)
    batch_patch_mask = torch.stack(batch[1], dim=0)
    return batch_pixels_values, batch_patch_mask
