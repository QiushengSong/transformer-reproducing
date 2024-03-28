import layers as L
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy


class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 self_attn,
                 feed_forward,
                 dropout: float=0.1,
                 ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = L.clones(L.SubLayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, mask=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))

        x = self.sublayer[1](x, self.feed_forward)

        return x


class DecoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 self_attn,
                 src_attn,
                 feed_forward,
                 dropout: float=0.1,
                 ):
        super().__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = L.clones(L.SubLayerConnection(d_model, dropout), 3)

    def forward(self,
                x,
                memory,
                src_mask,
                tgt_mask,
                ):
        m = memory
        # Query and value from the Encoder: used to decode the sequence;
        x = self.sublayer[0](x, lambda x:self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x:self.self_attn(x, m, m, src_mask))

        return self.sublayer[2](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self,
                 EncoderLayer,
                 num_of_EncoderLayer: int,
                 ):
        super().__init__()
        self.EncoderLayers = L.clones(EncoderLayer, num_of_EncoderLayer)
        self.layernorm = L.LayerNorm(EncoderLayer.d_model)

    def forward(self, x, mask=None):
        for encoderlayer in self.EncoderLayers:
            x = encoderlayer(x, mask)
            x = self.LayerNorm(x)

        return x


class Decoder(nn.Module):
    def __init__(self,
                 DecoderLayer,
                 N):
        super().__init__()
        self.DecoderLayers = L.clones(DecoderLayer, N)
        self.layernorm = L.LayerNorm(EncoderLayer.d_model)

    def forward(self,
                x,
                memory,
                src_mask,
                tgt_mask,
                ):
        for decoderlayer in self.DecoderLayers:
            x = decoderlayer(x, memory, src_mask, tgt_mask)
            x = self.layernorm(x)

        return x


class Generator(nn.Module):
    def __init__(self,
                 d_model: int,
                 vocab_size: int,
                 ):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, inputs):

        return F.log_softmax(self.linear(inputs), dim=-1)


class EncoderDecoder(nn.Module):
    def __init__(self,
                 Encoder,
                 Decoder,
                 src_embed,
                 tgt_embed,
                 generator
                 ):
        super().__init__()
        self.encoder = Encoder
        self.decoder = Decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        src = self.src_embed
        src = self.encoder(src, src_mask)
        embed_tgt = self.tgt_embed(tgt)
        Decoder_result = self.decoder(embed_tgt, src, src_mask, tgt_mask)
        output = self.generator(Decoder_result)
        return output