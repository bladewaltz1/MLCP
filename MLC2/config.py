from yacs.config import CfgNode as CN

_C = CN()
_C.save_dir = "output/"

_C.start_epoch = 0
_C.epochs = 400
_C.device = "cuda"
_C.log_time = 20
_C.ckpt_time = 20
_C.samples_per_gpu = 64
_C.num_workers = 8
_C.warmup_epoches = 0

_C.solver = CN()
_C.solver.lr = 1e-4
_C.solver.weight_decay = 5e-2
_C.solver.betas = (0.9, 0.97)

_C.temperature = CN()
_C.temperature.init_value = 1.0
_C.temperature.min_value = 0.0625
_C.temperature.epoches = 100

_C.hidden_size = 768
_C.image_size = 256
_C.patch_size = 16
_C.layer_norm_eps = 1e-12
_C.num_codes = 8192
_C.initializer_range = 0.02

_C.mlc_decoder_cfg = CN()
_C.mlc_decoder_cfg.num_queries = 100
_C.mlc_decoder_cfg.hidden_size = 256
_C.mlc_decoder_cfg.nhead = 8
_C.mlc_decoder_cfg.ffn_dim = 2048
_C.mlc_decoder_cfg.dropout = 0.0
_C.mlc_decoder_cfg.num_layers = 6

_C.pixel_decoder_cfg = CN()
_C.pixel_decoder_cfg.hidden_size = 256
_C.pixel_decoder_cfg.nhead = 8
_C.pixel_decoder_cfg.ffn_dim = 2048
_C.pixel_decoder_cfg.dropout = 0.0
_C.pixel_decoder_cfg.num_layers = 6
