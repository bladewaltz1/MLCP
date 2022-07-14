import sys
try:
    import hf_env
    hf_env.set_env('202111')
    sys.path.insert(0, '/ceph-jd/pub/jupyter/maoweian/notebooks/code/self-supervise/mae/env')
except:
    pass


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

from config import _C as cfg
from data import Dataset, collate_fn
from model import PretrainModel


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
            iteration = iteration + 1
            batch = [p.to(cfg.device) for p in batch]

            with torch.cuda.amp.autocast():
                loss_rec, loss_dvae, indices = model(*batch)
                loss = loss_rec * cfg.solver.rec_weight + \
                    loss_dvae * cfg.solver.dvae_weight

            loss_scaler(loss, optimizer, parameters=model.parameters())
            optimizer.zero_grad()
            scheduler.step()

            if iteration % cfg.log_time == 0:
                logger.info(
                    "  ".join([
                        "iter: {iter}", 
                        "loss_rec: {loss_rec:.4f}", 
                        "loss_dvae: {loss_dvae:.4f}",
                        "#indices: {num_indices}",
                        "lr: {lr:.8f}",
                    ]).format(
                        iter=iteration, 
                        loss_rec=loss_rec, 
                        loss_dvae=loss_dvae, 
                        num_indices=len(indices.unique()),
                        lr=optimizer.param_groups[0]["lr"],
                    ))

        checkpointer.save("model_{:04d}".format(epoch))
        checkpointer.save("last_checkpoint")


def main():
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--hfai_enable", default=False, type=bool,
                        help='whether use hfai cluster')

    parser.add_argument("--config-file", default="", metavar="FILE")
    parser.add_argument("--load-last-checkpoint", action="store_true")
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

    if args.hfai_enable:
        import pickle
        import torchvision.transforms as transforms

        transform = transforms.Compose([
            transforms.RandomResizedCrop(cfg.image_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        from ffrecord.torch import Dataset, DataLoader
        class FireFlyerImageNet(Dataset):
            def __init__(self, fnames, transform=None):
                super(FireFlyerImageNet, self).__init__(fnames, check_data=True)
                self.transform = transform

            def process(self, indexes, data):
                samples = []

                for bytes_ in data:
                    img, label = pickle.loads(bytes_)
                    if self.transform:
                        img = self.transform(img)
                    samples.append((img, label))

                # default collate_fn would handle them
                return samples
        dataset = FireFlyerImageNet('/public_dataset/1/ImageNet/train.ffr', transform=transform)
    else:
        dataset = Dataset(cfg)

    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=cfg.solver.lr,
                                  weight_decay=cfg.solver.weight_decay,
                                  betas=cfg.solver.betas)

    loss_scaler = NativeScalerWithGradNormCount()

    num_gpus = get_world_size()
    iterations_per_epoch = len(dataset) // (cfg.samples_per_gpu * num_gpus)
    warmup_steps = iterations_per_epoch * cfg.warmup_epoches
    max_steps = cfg.epochs * iterations_per_epoch
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps
    )

    checkpointer = Checkpointer(model=model,
                                optimizer=optimizer,
                                loss_scaler=loss_scaler,
                                scheduler=scheduler,
                                save_dir=save_dir,
                                save_to_disk=get_rank() == 0,
                                logger=logger)
    if args.load_last_checkpoint:
        path = os.path.joint(save_dir, "last_checkpoint.pth")
        if os.path.exists(path):
            checkpointer.load(path)


    if args.hfai_enable:
        if args.distributed:
            num_tasks = get_world_size()
            global_rank = get_rank()
            sampler_train = torch.utils.data.DistributedSampler(
                dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            print("Sampler_train = %s" % str(sampler_train))
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset)

        data_loader = DataLoader(dataset,
                                 cfg.samples_per_gpu,
                                 sampler=sampler_train,
                                 num_workers=cfg.num_workers,
                                 pin_memory=True,
                                 drop_last=True)
    else:
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


if __name__ == "__main__":
    from detectron2.engine import launch
    ip = os.environ['MASTER_IP']
    port = os.environ['MASTER_PORT']
    hosts = int(os.environ['WORLD_SIZE'])  # 机器个数
    rank = int(os.environ['RANK'])  # 当前机器编号
    gpus = torch.cuda.device_count()  # 每台机器的GPU个数

    launch(
        main,
        gpus,
        num_machines=hosts,
        machine_rank=rank,
        dist_url=f'tcp://{ip}:{port}',
    )
