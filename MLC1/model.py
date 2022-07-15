import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from transformers.models.clip.modeling_clip import (
    CLIPVisionTransformer, 
    CLIPVisionConfig, 
    CLIPTextTransformer, 
    CLIPTextConfig
)

from utils import patchify
from utils.transformer import TransformerDecoderLayer, TransformerDecoder


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def contrastive_loss(q, k, tau):
    # gather all targets
    world_size = 1
    rank = 0
    if torch.distributed.is_initialized():
        k = concat_all_gather(k)
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    # Einstein sum is more intuitive
    logits = torch.einsum("nc,mc->nm", [q, k]) / tau
    N = logits.shape[0]  # batch size per GPU
    S = int(logits.shape[1] / world_size / N)
    labels = (torch.arange(N, dtype=torch.long) + N * S * rank).cuda()
    return F.cross_entropy(logits, labels) / 2


class ImageEmbeddingsWithMask(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        embed_dim = cfg.pixel_decoder_cfg.embed_dim
        self.patch_embedding = nn.Conv2d(in_channels=3, 
                                         out_channels=embed_dim,
                                         kernel_size=cfg.patch_size, 
                                         stride=cfg.patch_size, 
                                         bias=False)

        num_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.position_embedding = nn.Embedding(num_patches, embed_dim)

        position_ids = torch.arange(num_patches).expand((1, -1))
        self.register_buffer("position_ids", position_ids)

        self.mask_embedding = nn.Parameter(torch.zeros([1, 1, embed_dim]))
        torch.nn.init.normal_(self.mask_embedding, std=.02)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, pixel_values, mask):
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        # patch_embeds[mask] = self.mask_embedding
        mask = mask.to(torch.float32).unsqueeze(-1)
        patch_embeds = patch_embeds * (1 - mask) + self.mask_embedding * mask

        embeddings = patch_embeds + self.position_embedding(self.position_ids)
        embeddings = self.layer_norm(embeddings)
        return embeddings


class TextEmbeddingsWithMask(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        embed_dim = cfg.token_decoder_cfg.embed_dim
        self.token_embedding = nn.Embedding(cfg.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(cfg.max_position_embeddings, 
                                               embed_dim)

        position_ids = torch.arange(cfg.max_position_embeddings).expand((1, -1))
        self.register_buffer("position_ids", position_ids)

        self.mask_embedding = nn.Parameter(torch.zeros([1, 1, embed_dim]))
        torch.nn.init.normal_(self.mask_embedding, std=.02)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids, mask):
        seq_length = input_ids.shape[-1]
        position_ids = self.position_ids[:, :seq_length]
        position_embeds = self.position_embedding(position_ids)

        inputs_embeds = self.token_embedding(input_ids)
        # inputs_embeds[mask] = self.mask_embedding
        mask = mask.to(torch.float32).unsqueeze(-1)
        inputs_embeds = inputs_embeds * (1 - mask) + self.mask_embedding * mask

        embeddings = inputs_embeds + position_embeds
        embeddings = self.layer_norm(embeddings)
        return embeddings


class MLCDecoder(nn.Module):
    def __init__(self, cfg, num_queries, projection_dim):
        super().__init__()
        self.mlc_embedding = nn.Embedding(num_queries, cfg.embed_dim)
        decoder_layer = TransformerDecoderLayer(cfg.embed_dim,
                                                cfg.nhead,
                                                cfg.ffn_dim,
                                                dropout=cfg.dropout,
                                                activation=F.gelu,
                                                batch_first=True,
                                                norm_first=False)
        self.decoder = TransformerDecoder(decoder_layer, cfg.num_layers)
        self.projection = nn.Linear(cfg.embed_dim, 
                                    projection_dim, 
                                    bias=False)

    def forward(self, encoder_hidden_states, pad_mask=None):
        bs = encoder_hidden_states.size(0)
        mlc_queries = self.mlc_embedding.weight[None]
        mlc_queries = mlc_queries.repeat(bs, 1, 1)
        mlc_queries, attn_weights = self.decoder(tgt=mlc_queries,
                                                 memory=encoder_hidden_states,
                                                 memory_key_padding_mask=pad_mask)

        projected = self.projection(mlc_queries)
        projected = projected / projected.norm(dim=-1, keepdim=True)

        return mlc_queries, projected, attn_weights


class DenoiseDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(cfg.embed_dim,
                                                cfg.nhead,
                                                cfg.ffn_dim,
                                                dropout=cfg.dropout,
                                                activation=F.gelu,
                                                batch_first=True,
                                                norm_first=False)
        self.decoder = TransformerDecoder(decoder_layer, cfg.num_layers)

    def forward(self, mlc_queries, masked_input, pad_mask=None):
        denoised_output, attn_weights = self.decoder(tgt=masked_input,
                                                     memory=mlc_queries,
                                                     tgt_key_padding_mask=pad_mask)
        return denoised_output, attn_weights


class PretrainModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        img_encoder_cfg = CLIPVisionConfig(image_size=cfg.image_size,
                                           patch_size=cfg.patch_size)
        self.img_encoder = CLIPVisionTransformer(img_encoder_cfg)
        txt_encoder_cfg = CLIPTextConfig(
            max_position_embeddings=cfg.max_position_embeddings
        )
        self.txt_encoder = CLIPTextTransformer(txt_encoder_cfg)

        self.img_decoder = MLCDecoder(cfg.img_decoder_cfg, 
                                      cfg.num_queries, 
                                      cfg.projection_dim)
        self.txt_decoder = MLCDecoder(cfg.txt_decoder_cfg,
                                      cfg.num_queries, 
                                      cfg.projection_dim)

        self.img_embedding = ImageEmbeddingsWithMask(cfg)
        self.txt_embedding = TextEmbeddingsWithMask(cfg)

        self.pixel_decoder = DenoiseDecoder(cfg.pixel_decoder_cfg)
        self.token_decoder = DenoiseDecoder(cfg.token_decoder_cfg)

        self.pixel_head = nn.Linear(cfg.pixel_decoder_cfg.embed_dim, 
                                    cfg.patch_size ** 2 * 3, 
                                    bias=True)
        self.token_head = nn.Linear(cfg.token_decoder_cfg.embed_dim,
                                    cfg.vocab_size,
                                    bias=True)

        temperature = cfg.logit_scale_init_value
        self.logit_scale = nn.Parameter(torch.tensor([temperature]))
        identity_mat = torch.diag(torch.ones(cfg.num_queries))[None]
        self.register_buffer("identity_mat", identity_mat)

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

    def forward(self, img, txt, pad_mask, img_mask, txt_mask, balance_weight):
        # image encoding
        output_ = self.img_encoder(img)
        img_hidden_states = output_.last_hidden_state
        img_hidden_states = self.img_encoder.post_layernorm(img_hidden_states)
        # image mlc decoding
        img_mlc, projected_img_mlc, _ = self.img_decoder(img_hidden_states)
        # orthogonal regularization
        identity_mat = self.identity_mat.repeat(img_mlc.size(0), 1, 1)
        img_reg = F.mse_loss(torch.bmm(projected_img_mlc,
                                       projected_img_mlc.transpose(2, 1)),
                             identity_mat)

        # text encoding
        output_ = self.txt_encoder(txt)
        txt_hidden_states = output_.last_hidden_state
        # text mlc decoding
        txt_mlc, projected_txt_mlc, _ = self.txt_decoder(txt_hidden_states, 
                                                         pad_mask)
        # orthogonal regularization
        txt_reg = F.mse_loss(torch.bmm(projected_txt_mlc,
                                       projected_txt_mlc.transpose(2, 1)),
                             identity_mat)

        # LSA between projected_img_query and projected_txt_mlc
        indices = []
        batch_cost = torch.bmm(projected_img_mlc, 
                               projected_txt_mlc.transpose(2, 1)).detach()
        for i, cost in enumerate(batch_cost):
            index = linear_sum_assignment(cost.cpu().numpy())
            index = torch.as_tensor(index[1], dtype=torch.int64)
            index += i * self.cfg.num_queries
            indices.append(index)
        indices = torch.cat(indices, dim=0)

        # contrastive loss between matched mlc
        projected_img_mlc = projected_img_mlc.view(-1, self.cfg.projection_dim)
        projected_txt_mlc = projected_txt_mlc.view(-1, self.cfg.projection_dim)
        projected_txt_mlc = projected_txt_mlc[indices, :]

        loss_contrastive = contrastive_loss(projected_img_mlc, 
                                            projected_txt_mlc,
                                            self.logit_scale) + \
                           contrastive_loss(projected_txt_mlc, 
                                            projected_img_mlc,
                                            self.logit_scale)

        # image reconstruction
        masked_img_embeds = self.img_embedding(img, img_mask)
        denoised_img_embeds, _ = self.pixel_decoder(txt_mlc, masked_img_embeds)
        denoised_img = self.pixel_head(denoised_img_embeds[img_mask])
        patches = patchify(img, self.cfg.patch_size)[img_mask]
        mean = patches.mean(dim=-1, keepdim=True)
        var = patches.var(dim=-1, keepdim=True)
        patches = (patches - mean) / (var + 1e-6) ** 0.5
        loss_img_reconstruction = F.mse_loss(denoised_img, patches)

        # caption reconstruction
        masked_txt_embeds = self.txt_embedding(txt, txt_mask)
        denoised_txt_embeds, _ = self.token_decoder(img_mlc, 
                                                    masked_txt_embeds, 
                                                    pad_mask)
        denoised_txt = self.token_head(denoised_txt_embeds[txt_mask])
        loss_txt_reconstruction = F.cross_entropy(denoised_txt, 
                                                  txt[txt_mask], 
                                                  weight=balance_weight)

        return {"loss_ctr": loss_contrastive, 
                "loss_reg": img_reg + txt_reg,
                "loss_imgrec": loss_img_reconstruction, 
                "loss_txtrec": loss_txt_reconstruction}
