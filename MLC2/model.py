import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.vit.modeling_vit import ViTEncoder, ViTConfig

from utils import patchify
from utils.transformer import TransformerDecoderLayer, TransformerDecoder


class ImageEmbeddings(nn.Module):
    def __init__(self, cfg, dim):
        super().__init__()
        self.patch_embedding = nn.Conv2d(in_channels=3, 
                                         out_channels=dim,
                                         kernel_size=cfg.patch_size, 
                                         stride=cfg.patch_size, 
                                         bias=False)

        num_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.position_embedding = nn.Embedding(num_patches, dim)
        position_ids = torch.arange(num_patches).expand((1, -1))
        self.register_buffer("position_ids", position_ids)

    def forward(self, pixel_values):
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        embeddings = patch_embeds + self.position_embedding(self.position_ids)
        return embeddings


class PositionEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        num_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.position_embedding = nn.Embedding(num_patches,
                                               cfg.pixel_decoder_cfg.hidden_size)
        position_ids = torch.arange(num_patches).expand((1, -1))
        self.register_buffer("position_ids", position_ids)

    def forward(self):
        embeddings = self.position_embedding(self.position_ids)
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


class GumbelQuantize(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding = nn.Embedding(cfg.num_codes, 
                                      cfg.mlc_decoder_cfg.hidden_size)

    def forward(self, mlc_emb, tau):
        logits = torch.einsum("b l d, c d -> b l c", 
                              mlc_emb, self.embedding.weight)
        soft_one_hot = F.gumbel_softmax(logits, 
                                        tau=tau, 
                                        dim=-1, 
                                        hard=False)
        quantized = torch.einsum("b l c, c d -> b l d", 
                                 soft_one_hot, 
                                 self.embedding.weight)
        return quantized, soft_one_hot.argmax(dim=-1)


class DenoiseDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(cfg.hidden_size,
                                                cfg.nhead,
                                                cfg.ffn_dim,
                                                dropout=cfg.dropout,
                                                activation=F.gelu,
                                                batch_first=True,
                                                norm_first=True)
        self.decoder = TransformerDecoder(decoder_layer, cfg.num_layers)
        self.layernorm = nn.LayerNorm(cfg.hidden_size, eps=1e-12)

    def forward(self, mlc_queries, masked_input):
        denoised_output, attn_weights = self.decoder(tgt=masked_input,
                                                     memory=mlc_queries)
        denoised_output = self.layernorm(denoised_output)
        return denoised_output, attn_weights


class PretrainModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_embedding = ImageEmbeddings(cfg, cfg.hidden_size)
        encoder_cfg = ViTConfig(image_size=cfg.image_size,
                                patch_size=cfg.patch_size)
        self.encoder = ViTEncoder(encoder_cfg)
        self.layernorm = nn.LayerNorm(cfg.hidden_size, eps=1e-12)
        self.projection = nn.Linear(cfg.hidden_size, 
                                    cfg.mlc_decoder_cfg.hidden_size)
        self.mlc_decoder = MLCDecoder(cfg.mlc_decoder_cfg)

        self.codebook = GumbelQuantize(cfg)
        self.position_embedding = PositionEmbeddings(cfg)
        self.pixel_decoder = DenoiseDecoder(cfg.pixel_decoder_cfg)
        self.pixel_head = nn.Linear(cfg.pixel_decoder_cfg.hidden_size, 
                                    cfg.patch_size ** 2 * 3, 
                                    bias=True)

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

    def forward(self, img, tau):
        bs = img.size(0)

        # image encoding
        patch_emb = self.patch_embedding(img)
        encoder_output = self.encoder(patch_emb)
        hidden_states = encoder_output.last_hidden_state
        hidden_states = self.layernorm(hidden_states)

        # image mlc decoding
        hidden_states = self.projection(hidden_states)
        mlc_emb, _ = self.mlc_decoder(hidden_states)

        # quantization
        quantized, indices = self.codebook(mlc_emb, tau)

        # image reconstruction
        position_embs = self.position_embedding().repeat(bs, 1, 1)
        denoised_patch_embs, _ = self.pixel_decoder(quantized, position_embs)
        denoised_patches = self.pixel_head(denoised_patch_embs)
        target_patches = patchify(img, self.cfg.patch_size)
        mean = target_patches.mean(dim=-1, keepdim=True)
        var = target_patches.var(dim=-1, keepdim=True)
        target_patches = (target_patches - mean) / (var + 1.0e-6) ** 0.5
        loss_rec = F.mse_loss(denoised_patches, target_patches)

        return loss_rec, indices
