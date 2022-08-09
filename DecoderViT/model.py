import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.vit.modeling_vit import ViTEncoder, ViTConfig

from utils.transformer import TransformerDecoderLayer, TransformerDecoder


class ImageEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_embedding = nn.Conv2d(in_channels=3, 
                                         out_channels=cfg.hidden_size,
                                         kernel_size=cfg.patch_size, 
                                         stride=cfg.patch_size, 
                                         bias=False)

        num_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.position_embedding = nn.Embedding(num_patches, cfg.hidden_size)
        position_ids = torch.arange(num_patches).expand((1, -1))
        self.register_buffer("position_ids", position_ids)

    def forward(self, pixel_values):
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        embeddings = patch_embeds + self.position_embedding(self.position_ids)
        return embeddings


class MLCDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlc_embedding = nn.Embedding(cfg.num_queries, cfg.hidden_size)
        decoder_layer = TransformerDecoderLayer(cfg.hidden_size,
                                                cfg.nhead,
                                                cfg.ffn_dim,
                                                dropout=cfg.dropout,
                                                activation=F.gelu,
                                                batch_first=True,
                                                norm_first=False)
        self.decoder = TransformerDecoder(decoder_layer, cfg.num_layers)

    def forward(self, encoder_hidden_states):
        bs = encoder_hidden_states.size(0)
        mlc_queries = self.mlc_embedding.weight[None]
        mlc_queries = mlc_queries.repeat(bs, 1, 1)
        mlc_queries, attn_weights = self.decoder(tgt=mlc_queries,
                                                 memory=encoder_hidden_states)
        return mlc_queries, attn_weights


class DecoderViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_embedding = ImageEmbeddings(cfg)
        encoder_cfg = ViTConfig(image_size=cfg.image_size,
                                patch_size=cfg.patch_size)
        self.encoder = ViTEncoder(encoder_cfg)
        self.decoder = MLCDecoder(cfg)
        self.layernorm = nn.LayerNorm(cfg.hidden_size, eps=1e-12)
        self.classifier = nn.Linear(cfg.hidden_size, cfg.num_classes)

        self.cfg = cfg
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.cfg.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.cfg.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, img):
        patch_emb = self.patch_embedding(img)
        encoder_output = self.encoder(patch_emb)
        hidden_states = encoder_output.last_hidden_state
        hidden_states = self.layernorm(patch_emb)
        mlc_emb, _ = self.decoder(hidden_states)
        return self.classifier(mlc_emb)
