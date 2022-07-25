import argparse
import logging
import os

import torch
import transformers
from torch.nn.parallel import DistributedDataParallel

from utils import get_rank, get_world_size, mkdir, synchronize
from utils.amp import NativeScalerWithGradNormCount
from utils.checkpointer import Checkpointer
from utils.dataloader import make_data_loader
from utils.logger import setup_logger

from MLC1.config import _C as cfg
from MLC1.data import Dataset, collate_fn
from MLC1.model import PretrainModel


def train(cfg, model, optimizer, loss_scaler, data_loader, 
          scheduler, checkpointer):
    logger = logging.getLogger("train")
    logger.info("Start training")
    model.train()

    for epoch in range(cfg.start_epoch, cfg.epochs):
        if cfg.distributed:
            data_loader.batch_sampler.sampler.set_epoch(epoch)
        optimizer.zero_grad()

        for iteration, batch in enumerate(data_loader):
            batch = [p.to(cfg.device) for p in batch]

            with torch.cuda.amp.autocast():
                loss_dict = model(*batch)
                loss = loss_dict["loss_img_ctr"] * cfg.solver.img_ctr_weight + \
                       loss_dict["loss_txt_ctr"] * cfg.solver.txt_ctr_weight + \
                       loss_dict["loss_img_rec"] * cfg.solver.img_rec_weight + \
                       loss_dict["loss_txt_rec"] * cfg.solver.txt_rec_weight + \
                       loss_dict["loss_img_reg"] * cfg.solver.img_reg_weight + \
                       loss_dict["loss_txt_reg"] * cfg.solver.txt_reg_weight

            loss_scaler(loss, optimizer, parameters=model.parameters())
            optimizer.zero_grad()
            scheduler.step()

            if iteration % cfg.log_time == 0:
                logger.info(
                    "  ".join([
                        "iter: {iter}", 
                        "loss_img_ctr: {loss_img_ctr:.4f}", 
                        "loss_txt_ctr: {loss_txt_ctr:.4f}", 
                        "loss_img_reg: {loss_img_reg:.4f}", 
                        "loss_txt_reg: {loss_txt_reg:.4f}", 
                        "loss_img_rec: {loss_img_rec:.4f}", 
                        "loss_txt_rec: {loss_txt_rec:.4f}",
                        "lr: {lr:.8f}",
                    ]).format(
                        iter=iteration, 
                        loss_img_ctr=loss_dict["loss_img_ctr"], 
                        loss_txt_ctr=loss_dict["loss_txt_ctr"], 
                        loss_img_reg=loss_dict["loss_img_reg"], 
                        loss_txt_reg=loss_dict["loss_txt_reg"], 
                        loss_img_rec=loss_dict["loss_img_rec"], 
                        loss_txt_rec=loss_dict["loss_txt_rec"], 
                        lr=optimizer.param_groups[0]["lr"],
                    ))
            iteration = iteration + 1

        checkpointer.save("model_{:04d}".format(epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--config-file", default="", metavar="FILE")
    parser.add_argument("--load-clip", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if cfg.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group("nccl", init_method="env://")
        synchronize()

    save_dir = os.path.join(cfg.save_dir, f"train")
    mkdir(save_dir)
    logger = setup_logger("train", save_dir, get_rank())
    logger.info("Running with cfg:\n{}".format(cfg))

    model = PretrainModel(cfg)
    model = model.to(cfg.device)

    dataset = Dataset(cfg, "train")

    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=cfg.solver.lr,
                                  weight_decay=cfg.solver.weight_decay,
                                  betas=cfg.solver.betas)

    loss_scaler = NativeScalerWithGradNormCount()

    num_gpus = get_world_size()
    iterations_per_epoch = len(dataset) // (cfg.samples_per_gpu * num_gpus)
    warmup_steps = iterations_per_epoch
    max_steps = cfg.epochs * iterations_per_epoch
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps
    )

    checkpointer = Checkpointer(model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                loss_scaler=loss_scaler,
                                save_dir=save_dir,
                                save_to_disk=get_rank() == 0,
                                logger=logger)

    if args.load_clip:
        checkpointer.load(cfg.pretrained_clip, is_strict=False)
    if args.resume:
        checkpointer.load(cfg.model_path)

    data_loader = make_data_loader(dataset=dataset,
                                   collate_fn=collate_fn,
                                   batch_size=cfg.samples_per_gpu,
                                   num_workers=cfg.num_workers,
                                   shuffle=True,
                                   is_distributed=cfg.distributed)

    if cfg.distributed:
        model = DistributedDataParallel(module=model,
                                        device_ids=[args.local_rank],
                                        output_device=args.local_rank)

    train(cfg=cfg,
          model=model,
          optimizer=optimizer,
          loss_scaler=loss_scaler,
          data_loader=data_loader,
          scheduler=scheduler,
          checkpointer=checkpointer)
