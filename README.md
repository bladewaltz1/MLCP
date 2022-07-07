# MLCP

python -m torch.distributed.launch --nproc_per_node=8 -m MLC1.pretrain --config-file MLC1/configs/base.yml save_dir output/mlc1-base/ samples_per_gpu 32
