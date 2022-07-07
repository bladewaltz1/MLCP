import hf_env
hf_env.set_env('202111')

import os
import time
import pickle
from pathlib import Path
import torch
import torchvision
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms, models

from ffrecord.torch import Dataset, DataLoader

import hfai
import hfai.nccl.distributed as dist
from torch.multiprocessing import Process

hfai.client.bind_hf_except_hook(Process)


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


def train(dataloader, model, criterion, optimizer, scheduler, epoch, local_rank,
          start_step, best_acc, save_path):
    model.train()
    for step, batch in enumerate(dataloader):
        if step < start_step:
            continue

        samples, labels = [x.cuda(non_blocking=True) for x in batch]
        outputs = model(samples)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if local_rank == 0:
            print(f'Epoch: {epoch}, Step: {step}, Loss: {loss.item()}')

        # 保存
        rank = torch.distributed.get_rank()
        if rank == 0 and hfai.receive_suspend_command():
            state = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'acc': best_acc,
                'epoch': epoch,
                'step': step + 1
            }
