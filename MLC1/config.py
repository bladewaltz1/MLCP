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
_C.solver.img_fctr = 1.0
_C.solver.txt_fctr = 1.0
_C.solver.img_cctr = 1.0
_C.solver.txt_cctr = 1.0
_C.solver.img_rec = 4.0
_C.solver.txt_rec = 1.0
_C.solver.img_reg = 1.0
_C.solver.txt_reg = 4.0

_C.clip_config = "openai/clip-vit-base-patch16"
_C.image_size = 256
_C.logit_scale_init_value = 0.07
_C.max_position_embeddings = 512
_C.num_queries = 100
_C.patch_mask_ratio = 0.9
_C.patch_size = 16
_C.projection_dim = 256
_C.token_mask_ratio = 0.9
_C.vocab_size = 49408
_C.initializer_range = 0.02
_C.balance_weight = 0.1
_C.label_smoothing = 0.1

_C.img_decoder_cfg = CN()
_C.img_decoder_cfg.num_queries = 100
_C.img_decoder_cfg.embed_dim = 256
_C.img_decoder_cfg.nhead = 8
_C.img_decoder_cfg.ffn_dim = 2048
_C.img_decoder_cfg.dropout = 0.0
_C.img_decoder_cfg.num_layers = 5

_C.txt_decoder_cfg = CN()
_C.txt_decoder_cfg.num_queries = 100
_C.txt_decoder_cfg.embed_dim = 256
_C.txt_decoder_cfg.nhead = 8
_C.txt_decoder_cfg.ffn_dim = 2048
_C.txt_decoder_cfg.dropout = 0.0
_C.txt_decoder_cfg.num_layers = 5

_C.pixel_decoder_cfg = CN()
_C.pixel_decoder_cfg.embed_dim = 256
_C.pixel_decoder_cfg.nhead = 8
_C.pixel_decoder_cfg.ffn_dim = 2048
_C.pixel_decoder_cfg.dropout = 0.0
_C.pixel_decoder_cfg.num_layers = 2

_C.token_decoder_cfg = CN()
_C.token_decoder_cfg.embed_dim = 256
_C.token_decoder_cfg.nhead = 8
_C.token_decoder_cfg.ffn_dim = 2048
_C.token_decoder_cfg.dropout = 0.0
_C.token_decoder_cfg.num_layers = 2
