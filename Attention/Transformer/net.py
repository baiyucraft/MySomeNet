import copy
import torch
from torch import nn
from torch.nn import functional as F


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def multi_head_attention_forward(query, key, value, num_heads, in_proj_weight, in_proj_bias, dropout_p, out_proj_weight,
                                 out_proj_bias, training=True):
    tgt_len, bsz, embed_dim = query.size()
    head_dim = embed_dim // num_heads
    q, k, v = None, None, None

    if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
        # self-attention
        # 集体作 w 然后通过 chunk 在最后一维
        q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

    elif key is value or torch.equal(key, value):
        # encoder-decoder attention
        _w = in_proj_weight[:embed_dim, :]
        _b = in_proj_bias[:embed_dim]
        q = F.linear(query, _w, _b)

        _w = in_proj_weight[embed_dim:, :]
        _b = in_proj_bias[embed_dim:]
        k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

    # 将 q k v
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    src_len = k.size(1)

    # sqrt dk
    scaling = float(head_dim) ** -0.5
    # QK^T / sqrt(dk)  三维的后两维矩阵相乘  (bsz * num_heads, tgt_len, src_len)
    attn_output_weights = torch.bmm(q, k.transpose(1, 2)) * scaling
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    # 与V相乘 (bsz * num_heads, tgt_len, head_dim)
    attn_output = torch.bmm(attn_output_weights, v)
    # 转换回原来的
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
    return attn_output, attn_output_weights.sum(dim=1) / num_heads


class MultiHeadAttention(nn.Module):
    """
    math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))

        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.bias_k = self.bias_v = None

        self.add_zero_attn = False

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value):
        return multi_head_attention_forward(query, key, value, self.num_heads, self.in_proj_weight, self.in_proj_bias,
                                            self.dropout, self.out_proj.weight, self.out_proj.bias,
                                            training=self.training)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, src):
        # multiHeadAttention
        src1, _ = self.multi_head_attention(src, src, src)
        # dropout + 残差连接, 接着LayerNorm
        src1 = self.dropout1(src1)
        src1 = src + src1
        src1 = self.norm1(src1)

        # Feed Forward
        src2 = self.linear1(src1)
        src2 = self.activation(src2)
        src2 = self.dropout(src2)
        src2 = self.linear2(src2)
        # dropout + 残差连接, 接着LayerNorm
        src2 = self.dropout2(src2)
        src2 = src1 + src2
        src2 = self.norm2(src2)
        return src2


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src):
        output = src
        for mod in self.layers:
            output = mod(output)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, n_head, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()

        self.multi_head_attention1 = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.multi_head_attention2 = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, tgt, memory):
        # masked multi head attention
        tgt1, _ = self.multi_head_attention1(tgt, tgt, tgt)
        tgt1 = tgt + self.dropout1(tgt1)
        tgt1 = self.norm1(tgt1)

        # multi head attention
        tgt2, _ = self.multi_head_attention2(tgt1, memory, memory)
        tgt2 = tgt1 + self.dropout2(tgt2)
        tgt2 = self.norm2(tgt2)

        # Feed Forward
        tgt3 = self.linear1(tgt2)
        tgt3 = self.activation(tgt3)
        tgt3 = self.dropout(tgt3)
        tgt3 = self.linear2(tgt3)
        tgt3 = tgt2 + self.dropout3(tgt3)
        tgt3 = self.norm3(tgt3)
        return tgt3


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory):
        output = tgt
        for mod in self.layers:
            output = mod(output, memory)
        if self.norm is not None:
            output = self.norm(output)
        return output


class Transformer(nn.Module):
    def __init__(self, d_model=512, n_head=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
                 dropout=0.1, activation="relu"):
        super(Transformer, self).__init__()

        encoder_layer = TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, n_head, dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.n_head = n_head

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt):
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return output


if __name__ == '__main__':
    ts = Transformer()
    src = torch.rand(10, 32, 512)
    tgt = torch.rand((20, 32, 512))
    print(ts)
    print(ts(src, tgt).shape)
