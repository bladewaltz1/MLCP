from yacs.config import CfgNode as CN

_C = CN()
_C.save_dir = ""
_C.model_path = ""
_C.pretrained_clip = ""

_C.start_epoch = 0
_C.epochs = 100
_C.device = "cuda"
_C.log_time = 20
_C.distributed = True
_C.samples_per_gpu = 32
_C.num_workers = 8

_C.solver = CN()
_C.solver.lr = 1e-4
_C.solver.weight_decay = 5e-2
_C.solver.betas = (0.9, 0.97)
_C.solver.ctr_weight = 1.0
_C.solver.imgrec_weight = 4.0
_C.solver.txtrec_weight = 1.0
_C.solver.reg_weight = 1.0

_C.clip_config = "openai/clip-vit-base-patch16"
_C.image_size = 336
_C.logit_scale_init_value = 0.07
_C.max_position_embeddings = 512
_C.num_queries = 100
_C.patch_mask_ratio = 0.75
_C.patch_size = 16
_C.projection_dim = 512
_C.token_mask_ratio = 0.75
_C.vocab_size = 49408
_C.initializer_range = 0.02
_C.balance_weight = 0.25

_C.img_decoder_cfg = CN()
_C.img_decoder_cfg.num_queries = 100
_C.img_decoder_cfg.embed_dim = 768
_C.img_decoder_cfg.nhead = 12
_C.img_decoder_cfg.ffn_dim = 3072
_C.img_decoder_cfg.dropout = 0.0
_C.img_decoder_cfg.num_layers = 3

_C.txt_decoder_cfg = CN()
_C.txt_decoder_cfg.num_queries = 100
_C.txt_decoder_cfg.embed_dim = 512
_C.txt_decoder_cfg.nhead = 8
_C.txt_decoder_cfg.ffn_dim = 2048
_C.txt_decoder_cfg.dropout = 0.0
_C.txt_decoder_cfg.num_layers = 3

_C.pixel_decoder_cfg = CN()
_C.pixel_decoder_cfg.embed_dim = 512
_C.pixel_decoder_cfg.nhead = 8
_C.pixel_decoder_cfg.ffn_dim = 2048
_C.pixel_decoder_cfg.dropout = 0.0
_C.pixel_decoder_cfg.num_layers = 2

_C.token_decoder_cfg = CN()
_C.token_decoder_cfg.embed_dim = 768
_C.token_decoder_cfg.nhead = 12
_C.token_decoder_cfg.ffn_dim = 3072
_C.token_decoder_cfg.dropout = 0.0
_C.token_decoder_cfg.num_layers = 2
