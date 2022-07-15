from yacs.config import CfgNode as CN

_C = CN()
_C.save_dir = ""
_C.model_path = ""
_C.data_dir = ""

_C.start_epoch = 0
_C.epochs = 400
_C.device = "cuda"
_C.log_time = 20
_C.distributed = True
_C.samples_per_gpu = 32
_C.num_workers = 8
_C.warmup_epoches = 2

_C.solver = CN()
_C.solver.lr = 2.5e-4
_C.solver.weight_decay = 5e-2
_C.solver.betas = (0.9, 0.97)
_C.solver.rec_weight = 1.0
_C.solver.dvae_weight = 1.0
_C.solver.commitment_cost = 0.25
_C.solver.rec_weight = 1.0
_C.solver.dvae_weight = 0.5

_C.hidden_size = 768
_C.image_size = 336
_C.patch_mask_ratio = 1.0
_C.patch_size = 16
_C.layer_norm_eps = 1e-12
_C.num_codes = 8192
_C.initializer_range = 0.02

_C.mlc_decoder_cfg = CN()
_C.mlc_decoder_cfg.num_queries = 100
_C.mlc_decoder_cfg.hidden_size = 768
_C.mlc_decoder_cfg.nhead = 12
_C.mlc_decoder_cfg.ffn_dim = 3072
_C.mlc_decoder_cfg.dropout = 0.0
_C.mlc_decoder_cfg.num_layers = 6 # TODO

_C.pixel_decoder_cfg = CN()
_C.pixel_decoder_cfg.hidden_size = 768
_C.pixel_decoder_cfg.nhead = 12
_C.pixel_decoder_cfg.ffn_dim = 3072
_C.pixel_decoder_cfg.dropout = 0.0
_C.pixel_decoder_cfg.num_layers = 2
