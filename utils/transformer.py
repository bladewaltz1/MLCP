import torch.nn as nn


class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        output = self.multihead_attn(x, mem, mem,
                                     attn_mask=attn_mask,
                                     key_padding_mask=key_padding_mask)
        x, attn_weights = output
        return self.dropout2(x), attn_weights

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                short_cut=True):

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x_, attn_weights = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + x_ if short_cut else x_
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x_, attn_weights = self._mha_block(x, memory, memory_mask, memory_key_padding_mask)
            x = x + x_ if short_cut else x_
            x = self.norm2(x)
            x = self.norm3(x + self._ff_block(x))

        return x, attn_weights


class TransformerDecoder(nn.TransformerDecoder):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = tgt

        attn_weights_all = []
        short_cut = False
        for mod in self.layers:
            x, attn_weights = mod(x, memory, tgt_mask=tgt_mask, 
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  short_cut=short_cut)
            short_cut = True
            attn_weights_all.append(attn_weights)

        if self.norm is not None:
            x = self.norm(x)
        return x, attn_weights_all
