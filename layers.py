import copy
import torch.nn as nn
import torch.nn.functional as F


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

def _get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu
    raise RuntimeError('Activation should be relu/gelu, not {}'.format(activation))

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoderLayer_QKV(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(TransformerEncoderLayer_QKV, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        super(TransformerEncoderLayer_QKV, self).__setstate__(state)
        if 'activation' not in state:
            state['activation'] = F.relu

    def forward(self, query, key, src, src_mask=None, src_key_padding_mask = None):
        src2 = self.self_attn(query, key, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder_QKV(nn.Module):
    __constant__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder_QKV, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, query, key, src, mask=None, src_key_padding_mask = None):
        output = src
        for mod in self.layers:
            output = mod(query, key, output, src_mask = mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output

def TranformerEncoder(heads=8, layers=1, channels = 64):
    encoder_layer = TransformerEncoderLayer_QKV(
        d_model = channels, nhead=heads, dim_feedforward=64, activation='gelu'
    )
    return TransformerEncoder_QKV(encoder_layer, num_layers=layers)

class SequentialLearning(nn.Module):
    def __init__(self, channels, nheads):
        super().__init__()
        self.time_layer = TranformerEncoder(nheads, layers=1, channels=channels)

    def forward(self, y, base_shape):
        B, channels, seq_len = base_shape
        if seq_len == 1:
            return y
        v = y.permute(2, 0, 1)
        y = self.time_layer(v,v,v).permute(1,2,0)
        return y