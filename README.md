# MLCP

python -m torch.distributed.launch --nproc_per_node=6 -m MLC1.pretrain --config-file MLC1/configs/base.yml save_dir output/mlc1-base/ image_size 256 img_decoder_cfg.num_layers 3 txt_decoder_cfg.num_layers 3 samples_per_gpu 32
