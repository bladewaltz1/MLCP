import torch.nn as nn
import torch.nn.functional as F


class TransformerDecoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, kdim, vdim, 
                 dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, 
                                               dropout=dropout, 
                                               batch_first=batch_first,
                                               **factory_kwargs)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, 
                                                kdim=kdim, vdim=vdim, 
                                                dropout=dropout, 
                                                batch_first=batch_first,
                                                **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_pad_mask=None, memory_key_pad_mask=None, 
                short_cut=True):
        x = tgt
        if self.norm_first:
            x_, attn_weights = self._mha_block(self.norm1(x), memory, 
                                               memory_mask,
                                               memory_key_pad_mask)
            x = x + x_ if short_cut else x_
            x = x + self._sa_block(self.norm2(x), tgt_mask, tgt_key_pad_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x_, attn_weights = self._mha_block(x, memory, 
                                               memory_mask, 
                                               memory_key_pad_mask)
            x = x + x_ if short_cut else x_
            x = self.norm1(x)
            x = self.norm2(x + self._sa_block(x, tgt_mask, tgt_key_pad_mask))
            x = self.norm3(x + self._ff_block(x))

        return x, attn_weights

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout(x)

    # multihead attention block
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        output = self.cross_attn(x, mem, mem,
                                 attn_mask=attn_mask,
                                 key_padding_mask=key_padding_mask)
        x, attn_weights = output
        return self.dropout(x), attn_weights

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout(x)


class TransformerDecoder(nn.TransformerDecoder):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = tgt

        attn_weights_all = []
        for mod in self.layers:
            x, attn_weights = mod(x, memory, tgt_mask=tgt_mask, 
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)
            attn_weights_all.append(attn_weights)

        if self.norm is not None:
            x = self.norm(x)
        return x, attn_weights_all
