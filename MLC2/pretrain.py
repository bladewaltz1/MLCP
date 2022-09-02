import argparse
import os

import torch
import transformers
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler

from utils import mkdir
from utils.logger import setup_logger
from utils.scheduler import CosineScheduler

from MLC2.config import _C as cfg
from MLC2.model import PretrainModel

# hfai modules
import hfai
import hfai.distributed as HfaiDist
from hfai.datasets import HfaiImageNet
from hfai.nn.parallel import DistributedDataParallel as HfaiDDP


def init_dist(local_rank):
    # init dist
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = os.getenv("MASTER_PORT", 1024)
    hosts = int(os.getenv("WORLD_SIZE", 1))  # number of nodes
    rank = int(os.getenv("RANK", 0))  # node id
    gpus = torch.cuda.device_count()  # gpus per node

    HfaiDist.init_process_group(
        backend="nccl", 
        init_method=f"tcp://{ip}:{port}", 
        world_size=hosts * gpus, 
        rank=rank * gpus + local_rank
    )
    torch.cuda.set_device(local_rank)

    return HfaiDist.get_rank(), HfaiDist.get_world_size()


def main(local_rank):
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--config-file", default="", metavar="FILE")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = os.path.join(cfg.save_dir, f"train")
    mkdir(save_dir)

    rank, world_size = init_dist(local_rank)

    logger = setup_logger("train", save_dir, rank)
    logger.info("Running with cfg:\n{}".format(cfg))

    torch.manual_seed(12345)
    model = PretrainModel(cfg)
    # model = hfai.nn.to_hfai(model, verbose=True)
    model = HfaiDDP(model.to(cfg.device), device_ids=[local_rank])

    transform = transforms.Compose([
        transforms.RandomResizedCrop(
            224, scale=(0.2, 1.0), 
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])
    dataset = HfaiImageNet(split="train", transform=transform)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = dataset.loader(cfg.samples_per_gpu, 
                                sampler=sampler, 
                                num_workers=cfg.num_workers, 
                                pin_memory=True)

    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=cfg.solver.lr,
                                  weight_decay=cfg.solver.weight_decay,
                                  betas=cfg.solver.betas)

    steps_per_epoch = len(dataset) // (cfg.samples_per_gpu * world_size)
    warmup_steps = steps_per_epoch * cfg.warmup_epoches
    max_steps = cfg.epochs * steps_per_epoch
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps
    )
    temperature = CosineScheduler(
        cfg.temperature.init_value,
        cfg.temperature.min_value,
        cfg.temperature.epoches * steps_per_epoch
    )

    ckpt_path = os.path.join(save_dir, "latest.pth")
    start_epoch, start_step, _ = hfai.checkpoint.init(model=model, 
                                                      optimizer=optimizer, 
                                                      scheduler=scheduler,
                                                      ckpt_path=ckpt_path)
    temperature._step = start_epoch * steps_per_epoch + start_step

    model.train()
    for epoch in range(start_epoch, cfg.epochs):
        sampler.set_epoch(epoch)
        dataloader.set_step(start_step)

        for step, batch in enumerate(dataloader, start=start_step):
            batch_img = batch[0].to(cfg.device)

            temperature.step()
            loss_rec, code_id = model(batch_img, tau=temperature.value)

            optimizer.zero_grad()
            loss_rec.backward()
            optimizer.step()
            scheduler.step()

            model.try_save(epoch, step + 1)

            if step % cfg.log_time == 0:
                logger.info(
                    "  ".join([
                        "iter: {iter}", 
                        "loss_rec: {loss_rec:.4f}", 
                        "#indices: {num_indices}",
                        "tau: {tau:.8f}",
                        "lr: {lr:.8f}",
                    ]).format(
                        iter=step, 
                        loss_rec=loss_rec, 
                        tau=temperature.value, 
                        num_indices=len(code_id.unique()),
                        lr=optimizer.param_groups[0]["lr"],
                    )
                )

        if rank == 0 and (epoch % cfg.ckpt_time == 0 or epoch == cfg.epochs - 1):
            state = {"model": model.module.state_dict(), "epoch": epoch}
            torch.save(state, os.path.join(save_dir, f"{epoch:04d}.pth"))


if __name__ == "__main__":
    ngpus = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(), nprocs=ngpus)
