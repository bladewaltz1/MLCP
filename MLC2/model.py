import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from transformers.models.vit.modeling_vit import ViTEncoder, ViTConfig

from utils import patchify
from utils.transformer import TransformerDecoderLayer, TransformerDecoder


class ImageEmbeddingsWithMask(nn.Module):
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

        self.mask_embedding = nn.Parameter(torch.zeros([1, 1, cfg.hidden_size]))
        torch.nn.init.normal_(self.mask_embedding, std=0.02)

    def forward(self, pixel_values, mask=None):
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        if mask is not None:
            # patch_embeds[mask] = self.mask_embedding
            mask = mask.to(torch.float32).unsqueeze(-1)
            patch_embeds = patch_embeds * (1 - mask) + \
                           self.mask_embedding * mask

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


class CodeBook(nn.Module):
    def __init__(self, cfg):
        super(CodeBook, self).__init__()
        self.embedding = nn.Embedding(cfg.num_codes, cfg.hidden_size)
        self.commitment_cost = cfg.solver.commitment_cost

    def forward(self, mlc_emb):
        bs, seq_len, hidden_size = mlc_emb.shape
        mlc_emb = mlc_emb.view(-1, hidden_size)
        distances = (torch.sum(mlc_emb**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(mlc_emb, self.embedding.weight.t())).sqrt()

        distances = distances.view(bs, seq_len, -1)
        indices = [
            linear_sum_assignment(d.detach().cpu().numpy())[1]
            for d in distances
        ]
        indices = torch.from_numpy(np.concatenate(indices))
        indices = indices.to(mlc_emb.device)
        # indices = torch.min(distances, dim=-1)[1]
        # print("#indices: ", len(indices.unique()))

        quantized = self.embedding(indices)

        q_latent_loss = F.mse_loss(mlc_emb.detach(), quantized)
        e_latent_loss = F.mse_loss(mlc_emb, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized = mlc_emb + (quantized - mlc_emb).detach()

        return quantized.view(bs, seq_len, hidden_size), loss, indices


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
        self.patch_embedding = ImageEmbeddingsWithMask(cfg)
        encoder_cfg = ViTConfig(image_size=cfg.image_size,
                                patch_size=cfg.patch_size)
        self.encoder = ViTEncoder(encoder_cfg)
        self.layernorm = nn.LayerNorm(cfg.hidden_size, eps=1e-12)
        self.mlc_decoder = MLCDecoder(cfg.mlc_decoder_cfg)
        self.codebook = CodeBook(cfg)
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

    def forward(self, img):
        # image encoding
        patch_emb = self.patch_embedding(img)
        encoder_output = self.encoder(patch_emb)
        hidden_states = encoder_output.last_hidden_state
        hidden_states = self.layernorm(hidden_states)

        # image mlc decoding
        mlc_emb, _ = self.mlc_decoder(hidden_states)

        # quantization
        quantized, loss_dvae, indices = self.codebook(mlc_emb)

        # image reconstruction
        mask = mlc_emb.new_ones(patch_emb.shape[:2]).bool()
        masked_patch_embs = self.patch_embedding(img, mask)
        denoised_patch_embs, _ = self.pixel_decoder(quantized, masked_patch_embs)
        denoised_patches = self.pixel_head(denoised_patch_embs[mask])
        target_patches = patchify(img, self.cfg.patch_size)[mask]
        mean = target_patches.mean(dim=-1, keepdim=True)
        var = target_patches.var(dim=-1, keepdim=True)
        target_patches = (target_patches - mean) / (var + 1.0e-6) ** 0.5
        loss_reconstruction = F.mse_loss(denoised_patches, target_patches)

        return loss_reconstruction, loss_dvae, indices
