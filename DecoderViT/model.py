import copy

import torch
import torch.nn.functional as F
from torch import nn

from utils.transformer import TransformerDecoderLayer, TransformerDecoder


class ImageEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_embedding = nn.Conv2d(in_channels=3, 
                                         out_channels=cfg.enc_hidden_size,
                                         kernel_size=cfg.patch_size, 
                                         stride=cfg.patch_size, 
                                         bias=False)

        num_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.position_embedding = nn.Embedding(num_patches, cfg.enc_hidden_size)
        position_ids = torch.arange(num_patches).expand((1, -1))
        self.register_buffer("position_ids", position_ids)

    def forward(self, pixel_values):
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        embeddings = patch_embeds + self.position_embedding(self.position_ids)
        return embeddings


class DecoderViTBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(cfg.enc_hidden_size,
                                               cfg.enc_nhead,
                                               dropout=cfg.enc_dropout,
                                               batch_first=False,
                                               norm_first=True)
        self.ffn = nn.Sequential(nn.Linear(cfg.enc_hidden_size,
                                           cfg.enc_ffndim),
                                 nn.GELU(),
                                 nn.Dropout(cfg.enc_dropout),
                                 nn.Linear(cfg.enc_ffndim,
                                           cfg.enc_hidden_size))
        self.cross_attn = nn.MultiheadAttention(cfg.enc_hidden_size,
                                               cfg.enc_nhead,
                                               dropout=cfg.enc_dropout,
                                               kdim=cfg.dec_hidden_size,
                                               vdim=cfg.dec_hidden_size,
                                               batch_first=True,
                                               norm_first=True)
        self.dropout = nn.Dropout(cfg.enc_dropout)
        self.norm1 = nn.LayerNorm(cfg.enc_hidden_size, eps=cfg.layer_norm_eps)
        self.norm2 = nn.LayerNorm(cfg.enc_hidden_size, eps=cfg.layer_norm_eps)
        self.norm3 = nn.LayerNorm(cfg.enc_hidden_size, eps=cfg.layer_norm_eps)

        decoder_layer = TransformerDecoderLayer(cfg.dec_hidden_size,
                                                cfg.dec_nhead,
                                                cfg.dec_ffn_dim,
                                                kdim=cfg.enc_hidden_size,
                                                vdim=cfg.enc_hidden_size,
                                                dropout=cfg.dec_dropout,
                                                activation=F.gelu,
                                                batch_first=True,
                                                norm_first=False)
        self.decoder_layers = TransformerDecoder(decoder_layer, 
                                                 cfg.num_layers_per_block)

    def forward(self, patch_emb, query):
        """
        Args:
            patch_emb: (B, L, D)
            query: (B, l, D), normalized
        Returns:
            patch_emb: (B, L, D)
            query: (B, l, D), normalized
            attn_weights
        """
        x = patch_emb
        x_ = self.norm1(x)
        x_ = self.self_attn(x_, x_, x_, need_weights=False)[0]
        x_ = self.dropout(x_)
        x = x + x_
        x_ = self.norm2(x)
        x_ = self.ffn(x_)
        x_ = self.dropout(x_)
        x = x + x_
        x_ = self.norm3(x)

        query, attn_weights = self.decoder_layers(tgt=query, memory=x_)

        x_ = self.cross_attn(x_, query, query, need_weights=False)[0]
        x_ = self.dropout(x_)
        patch_emb = x + x_

        return patch_emb, query, attn_weights


class DecoderViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_embedding = ImageEmbeddings(cfg)
        self.query_embedding = nn.Embedding(cfg.num_queries, cfg.dec_hidden_size)
        block = DecoderViTBlock(cfg)
        self.backbone = nn.ModuleList(_get_clones(block, cfg.num_blocks))
        self.layernorm = nn.LayerNorm(cfg.enc_hidden_size, cfg.layer_norm_eps)
        self.classifier = nn.Linear(cfg.enc_hidden_size, cfg.num_classes)

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
        bs = img.size(0)
        patch_emb = self.patch_embedding(img)
        query = self.query_embedding.weight[None]
        query = query.repeat(bs, 1, 1)

        attns = []
        for block in self.backbone:
            patch_emb, query, attn = block(patch_emb, query)
            attns.append(attn)

        patch_emb = self.layernorm(patch_emb)
        avg_patch_emb = patch_emb.mean(dim=1)
        logits = self.classifier(avg_patch_emb)

        return logits, attns


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
