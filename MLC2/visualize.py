import argparse
import os
from PIL import Image

import torch
import numpy as np
import torch.nn.functional as F

from utils import mkdir, patchify, unpatchify
from utils.checkpointer import Checkpointer
from utils.dataloader import make_data_loader
from utils.logger import setup_logger

from MLC2.config import _C as cfg
from MLC2.data import Dataset, collate_fn
from MLC2.model import PretrainModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--config-file", default="", metavar="FILE")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = os.path.join(cfg.save_dir, f"visualize")
    mkdir(save_dir)
    logger = setup_logger("train", save_dir)
    logger.info("Running with cfg:\n{}".format(cfg))

    model = PretrainModel(cfg)
    model = model.to(cfg.device)

    dataset = Dataset(cfg)

    checkpointer = Checkpointer(model=model)
    checkpointer.load(cfg.model_path, model_only=True)

    data_loader = make_data_loader(dataset=dataset,
                                   collate_fn=collate_fn,
                                   batch_size=cfg.samples_per_gpu,
                                   num_workers=cfg.num_workers,
                                   shuffle=False,
                                   is_distributed=False)

    for iteration, batch in enumerate(data_loader):
        img = batch[0].to(cfg.device)
        mask = batch[1].to(cfg.device)

        with torch.no_grad():
            patch_emb = model.patch_embedding(img)
            encoder_output = model.encoder(patch_emb)
            hidden_states = encoder_output.last_hidden_state
            hidden_states = model.layernorm(hidden_states)

            mlc_emb, patch_attn_weights = model.mlc_decoder(hidden_states)
            quantized, loss_dvae, indices = model.codebook(mlc_emb)

            masked_patch_embs = model.patch_embedding(img, mask)
            denoised_patch_embs, mlc_attn_weights = model.pixel_decoder(
                quantized, masked_patch_embs
            )
            denoised_patches = model.pixel_head(denoised_patch_embs[mask])
            target_patches = patchify(img, cfg.patch_size)
            pd = target_patches.clone()
            gt = target_patches.clone()
            target_patches = target_patches[mask]
            mean = target_patches.mean(dim=-1, keepdim=True)
            var = target_patches.var(dim=-1, keepdim=True)
            target_patches = (target_patches - mean) / (var + 1.0e-6) ** 0.5
            loss_reconstruction = F.mse_loss(denoised_patches, target_patches)

            pd[mask] = denoised_patches
            pd = unpatchify(pd, cfg.patch_size)
            gt[mask] = target_patches
            gt = unpatchify(gt, cfg.patch_size)

        output = np.zeros([3, 2 * img.shape[2], img.shape[0] * img.shape[3]])
        for i, (src, tgt) in enumerate(zip(pd, gt)):
            src = src.cpu().numpy()
            tgt = tgt.cpu().numpy()
            output[:, :img.shape[2], i * img.shape[3]:(i + 1) * img.shape[3]] = tgt
            output[:, img.shape[2]:, i * img.shape[3]:(i + 1) * img.shape[3]] = src

        image_std = np.asarray(dataset.transform.image_std)
        image_mean = np.asarray(dataset.transform.image_mean)
        output = output * image_std[:, None, None] + \
                          image_mean[:, None, None]
        output = (output * 255.0).clip(0, 255)
        output = output.transpose(1, 2, 0)
        output = Image.fromarray(np.uint8(output), "RGB")
        output.save(os.path.join(save_dir, f"{iteration}__.jpg"))

        logger.info(
            "  ".join([
                "iter: {iteration}", 
                "loss_rec: {loss_reconstruction:.4f}", 
                "loss_dvae: {loss_dvae:.4f}",
            ]).format(
                iteration=iteration, 
                loss_reconstruction=loss_reconstruction, 
                loss_dvae=loss_dvae, 
            )
        )
