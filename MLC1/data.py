import itertools
import json
import os
import pickle
import random
from PIL import Image

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import CLIPFeatureExtractor, CLIPTokenizer


train_datasets = {
    "mscoco": {
        "ann_paths": [
            "datasets/coco_train_captions.jsonl",
            "datasets/coco_val_captions.jsonl",
            "datasets/coco/annotations/captions_train2017.json",
        ],
        "img_dirs": [
            "datasets/coco/train2017/",
            "datasets/coco/val2017/",
            "datasets/coco/train2017/",
        ]
    },
    "flickr30k": {
        "ann_paths": [
            "datasets/flickr30k_train_captions.jsonl",
            "datasets/flickr30k_val_captions.jsonl",
            "datasets/Flickr30k/results_20130124.token"
        ],
        "img_dirs": [
            "datasets/Flickr30k/flickr30k-images/",
            "datasets/Flickr30k/flickr30k-images/",
            "datasets/Flickr30k/flickr30k-images/",
        ]
    },
    "openimage": { 
        "ann_paths": [
            "datasets/open_images_train_v6_captions.jsonl",
            "datasets/open_images_validation_captions.jsonl",
            "datasets/open_images_test_captions.jsonl",
        ],
        "img_dirs": [
            "datasets/OpenImage/train/",
            "datasets/OpenImage/validation/",
            "datasets/OpenImage/test/",
        ]
    }, 
    "ade20k": { 
        "ann_paths": [
            "datasets/ade20k_train_captions.jsonl",
            "datasets/ade20k_validation_captions.jsonl",
        ],
        "img_dirs": [
            "datasets/ADE20k/",
            "datasets/ADE20k/",
        ],
        "index_file": "datasets/ADE20k/ADE20K_2021_17_01/index_ade20k.pkl"
    },
    "visualparagraph": {
        "ann_paths": [
            "datasets/VisualParagraph/paragraphs_v1.json"
        ],
        "img_dirs": [
            "datasets/VisualParagraph/VG_100K/"
        ]
    }
}

validation_datasets = {
    "mscoco": {
        "ann_paths": [
            "datasets/coco/annotations/captions_val2017.json"
        ],
        "img_dirs": [
            "datasets/coco/val2017/",
        ]
    },
    "flickr30k": {
        "ann_paths": [
            "datasets/flickr30k_test_captions.jsonl",
        ],
        "img_dirs": [
            "datasets/Flickr30k/flickr30k-images/",
        ]
    },
}


def build_openimage(ann_paths, img_dirs):
    output = []
    for ann_path, img_dir in zip(ann_paths, img_dirs):
        with open(ann_path) as f:
            f = list(f)
        for item in f:
            item = json.loads(item)
            image_id = item["image_id"]
            image_name = image_id + ".jpg"
            image_path = os.path.join(img_dir, image_name)
            caption = item["caption"].strip()
            output.append({"image_path": image_path,
                           "caption": [caption]})
    return output


def build_ade20k(ann_paths, img_dirs, index_file):
    output = []
    with open(index_file, "rb") as f:
        index_file = pickle.load(f)
    filename = index_file["filename"]
    folder = index_file["folder"]

    for ann_path, img_dir in zip(ann_paths, img_dirs):
        with open(ann_path) as f:
            f = list(f)
        for item in f:
            item = json.loads(item)
            image_id = item["image_id"]
            image_name = image_id + ".jpg"
            index = filename.index(image_name)
            image_path = os.path.join(img_dir, folder[index], image_name)
            caption = item["caption"].strip()
            output.append({"image_path": image_path,
                           "caption": [caption]})
    return output


def build_flickr30k(ann_paths, img_dirs):
    output = {}
    for ann_path, img_dir in zip(ann_paths, img_dirs):
        with open(ann_path) as f:
            if "jsonl" in ann_path:
                f = list(f)
                for item in f:
                    item = json.loads(item)
                    image_id = item["image_id"]
                    image_name = image_id + ".jpg"
                    image_path = os.path.join(img_dir, image_name)
                    caption = item["caption"].strip()
                    if image_id not in output.keys():
                        output[image_id] = {"image_path": image_path,
                                            "caption": [caption]}
                    else:
                        output[image_id]["caption"].append(caption)
            else:
                for line in f.readlines():
                    item = line.split("#")
                    image_id = item[0][:-4]
                    image_name = image_id + ".jpg"
                    image_path = os.path.join(img_dir, image_name)
                    caption = item[1][1:].strip()
                    if image_id not in output.keys():
                        output[image_id] = {"image_path": image_path,
                                            "caption": [caption]}
                    else:
                        output[image_id]["caption"].append(caption)

    return list(output.values())


def build_mscoco(ann_paths, img_dirs):
    output = {}
    for ann_path, img_dir in zip(ann_paths, img_dirs):
        with open(ann_path) as f:
            if "jsonl" in ann_path:
                f = list(f)
                for item in f:
                    item = json.loads(item)
                    image_id = item["image_id"]
                    image_name = "{:012d}.jpg".format(int(image_id))
                    image_path = os.path.join(img_dir, image_name)
                    caption = item["caption"].strip()
                    if image_id not in output.keys():
                        output[image_id] = {"image_path": image_path,
                                            "caption": [caption]}
                    else:
                        output[image_id]["caption"].append(caption)
            else:
                database = json.load(f)
                database = database['annotations']
                for item in database:
                    image_id = item["image_id"]
                    image_name = "{:012d}.jpg".format(image_id)
                    image_path = os.path.join(img_dir, image_name)
                    caption = item["caption"].strip()
                    if str(image_id) not in output.keys():
                        output[str(image_id)] = {"image_path": image_path,
                                                 "caption": [caption]}
                    else:
                        output[str(image_id)]["caption"].append(caption)

    return list(output.values())


def build_visualparagraph(ann_paths, img_dirs):
    output = []
    for ann_path, img_dir in zip(ann_paths, img_dirs):
        with open(ann_path) as f:
            database = json.load(f)
        for item in database:
            image_id = item["image_id"]
            image_name = str(image_id) + ".jpg"
            image_path = os.path.join(img_dir, image_name)
            caption = item["paragraph"].strip()
            output.append({"image_path": image_path,
                           "caption": [caption]})
    return output


class Dataset(Dataset):
    def __init__(self, cfg, split):
        datasets = globals()[f"{split}_datasets"]
        database = [globals()[f"build_{name}"](**kargs) 
                    for name, kargs in datasets.items()]
        self.database = list(itertools.chain(*database))
        self.transform = CLIPFeatureExtractor(
            do_resize=True,
            size=(cfg.image_size, cfg.image_size),
            do_center_crop=False,
            do_normalize=True
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.clip_config)
        self.tokenizer.model_max_length = cfg.max_position_embeddings
        self.cfg = cfg

    def __getitem__(self, index):
        image = Image.open(self.database[index]["image_path"])
        image = image.convert("RGB")
        pixels = self.transform(image, return_tensors="pt").pixel_values

        num_patches = (self.cfg.image_size // self.cfg.patch_size) ** 2
        num_masked_patches = int(num_patches * self.cfg.patch_mask_ratio)
        _, masked_indices = torch.rand(num_patches).topk(num_masked_patches)
        patch_mask = torch.zeros(num_patches, dtype=torch.long)
        patch_mask[masked_indices] = 1

        text = random.sample(self.database[index]["caption"], 1)
        tokens = self.tokenizer(text, return_tensors="pt").input_ids.squeeze(0)

        num_tokens = len(tokens)
        num_masked_tokens = int(num_tokens * self.cfg.token_mask_ratio)
        _, masked_indices = torch.rand(num_tokens).topk(num_masked_tokens)
        token_mask = torch.zeros(num_tokens, dtype=torch.long)
        token_mask[masked_indices] = 1
        token_mask[0] = 0
        token_mask[-1] = 0

        return pixels, tokens, patch_mask.bool(), token_mask.bool()

    def __len__(self):
        return len(self.database)


def collate_fn(batch):
    batch = list(zip(*batch))

    batch_pixels = torch.cat(batch[0], dim=0)
    batch_tokens = pad_sequence(batch[1], batch_first=True, padding_value=-1)
    padding_mask = batch_tokens == -1
    batch_tokens[padding_mask] = batch[1][0][-1]
    batch_patch_mask = torch.stack(batch[2], dim=0)
    batch_token_mask = pad_sequence(batch[3], batch_first=True, padding_value=0)

    return batch_pixels, batch_tokens, padding_mask, batch_patch_mask, \
           batch_token_mask
