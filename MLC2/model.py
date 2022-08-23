from audioop import mul
import multiprocessing

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from transformers.models.vit.modeling_vit import ViTEncoder, ViTConfig

from utils import patchify
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


class MLCModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_embedding = ImageEmbeddings(cfg)
        encoder_cfg = ViTConfig(image_size=cfg.image_size,
                                patch_size=cfg.patch_size)
        self.encoder = ViTEncoder(encoder_cfg)
        self.layernorm = nn.LayerNorm(cfg.hidden_size, eps=1e-12)
        self.projection = nn.Linear(cfg.hidden_size, 
                                     cfg.mlc_decoder_cfg.hidden_size)
        self.mlc_decoder = MLCDecoder(cfg.mlc_decoder_cfg)

    def forward(self, img):
        patch_emb = self.patch_embedding(img)
        encoder_output = self.encoder(patch_emb)
        hidden_states = encoder_output.last_hidden_state
        hidden_states = self.layernorm(hidden_states)

        hidden_states = self.projection(hidden_states)
        mlc_emb, attn = self.mlc_decoder(hidden_states)
        return mlc_emb, attn


class CodebookPre(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.projection = nn.Linear(cfg.mlc_decoder_cfg.hidden_size, 
                                    cfg.code_dim)

    def forward(self, code, mlc_emb):
        mlc_proj = self.projection(mlc_emb)
        mlc_proj = F.normalize(mlc_proj, p=2.0, dim=-1)
        similarity = torch.einsum("bld, cd -> blc", mlc_proj, code)
        distances = 1 - similarity
        return distances, mlc_proj


class CodebookPost(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.projection = nn.Linear(cfg.code_dim, 
                                    cfg.pixel_decoder_cfg.hidden_size)
        self.commitment_cost = cfg.solver.commitment_cost
        self.cfg = cfg

    def forward(self, mlc_proj, code, code_id):
        quantized = code[code_id] # bs, L, code_dim

        similarity = (mlc_proj * quantized).sum(dim=-1) # bs, L
        max_sim = similarity.topk(5, dim=-1)[0][:, [-1]]
        thresh = torch.minimum(max_sim, torch.ones_like(max_sim) * 
                                        self.cfg.threshold)
        valid = similarity >= thresh

        mlc_proj_valid = mlc_proj[valid]
        quantized_valid = quantized[valid]

        q_latent_loss = F.mse_loss(mlc_proj_valid.detach(), quantized_valid)
        e_latent_loss = F.mse_loss(mlc_proj_valid, quantized_valid.detach())
        loss_dvae = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = mlc_proj + (quantized - mlc_proj).detach()
        quantized = self.projection(quantized)

        return quantized, valid, loss_dvae


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

    def forward(self, mlc_queries, input, pad_mask=None):
        denoised_output, attn_weights = self.decoder(
            tgt=input,
            memory=mlc_queries,
            memory_key_padding_mask=pad_mask
        )
        denoised_output = self.layernorm(denoised_output)
        return denoised_output, attn_weights


class DenoiseHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.position_embedding = PositionEmbeddings(cfg)
        self.pixel_decoder = DenoiseDecoder(cfg.pixel_decoder_cfg)
        self.pixel_head = nn.Linear(cfg.pixel_decoder_cfg.hidden_size, 
                                    cfg.patch_size ** 2 * 3, 
                                    bias=True)
        self.cfg = cfg

    def forward(self, img, quantized, pad_mask):
        bs = quantized.size(0)
        position_embs = self.position_embedding().repeat(bs, 1, 1)
        denoised_patch_embs, attn = self.pixel_decoder(quantized, 
                                                       position_embs,
                                                       pad_mask)
        denoised_patches = self.pixel_head(denoised_patch_embs)

        target_patches = patchify(img, self.cfg.patch_size)
        mean = target_patches.mean(dim=-1, keepdim=True)
        var = target_patches.var(dim=-1, keepdim=True)
        target_patches = (target_patches - mean) / (var + 1.0e-6) ** 0.5
        loss_rec = F.mse_loss(denoised_patches, target_patches)

        return loss_rec, attn


class Pretrain(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.codebook = nn.Embedding(cfg.num_codes, cfg.code_dim)
        self.mlc_model = MLCModel(cfg)
        self.codebookpre = CodebookPre(cfg)
        self.codebookpost = CodebookPost(cfg)
        self.denoise_head = DenoiseHead(cfg)
        self.cfg = cfg
        self.apply(self._init_weights)

        self.pool =  multiprocessing.Pool()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.cfg.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.cfg.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, img):
        code = self.codebook.weight # num_codes, code_dim
        code = F.normalize(code, p=2.0, dim=-1)
        mlc_emb, _ = self.mlc_model(img)
        distances, mlc_proj = self.codebookpre(code, mlc_emb)

        distances = distances.detach().cpu().numpy() # bs, L, code_dim
        code_id = self.pool.map(lsa, list(distances)) # bs * (L,)
        code_id = torch.from_numpy(np.stack(code_id))
        code_id = code_id.to(mlc_proj.device)

        quantized, mask, loss_dvae = self.codebookpost(mlc_proj, code, code_id)
        loss_rec, _ = self.denoise_head(img, quantized, ~mask)
        return loss_rec, loss_dvae, code_id, mask


def lsa(mat):
    return linear_sum_assignment(mat)[1]
