import torch


def make_data_loader(dataset, batch_size, num_workers, shuffle,
                     collate_fn=None, is_distributed=False):
    if is_distributed:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=shuffle)
    elif shuffle:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    batch_sampler = torch.utils.data.BatchSampler(sampler, 
                                                  batch_size, 
                                                  drop_last=True)
    data_loader = torch.utils.data.DataLoader(dataset, 
                                              num_workers=num_workers,
                                              batch_sampler=batch_sampler, 
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader
