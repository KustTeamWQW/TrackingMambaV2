import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class TemporalAggregationDecoder(nn.Module):
    def __init__(self, d_model=384, nhead=8, num_decoder_layers=3,
                 dim_feedforward=384, dropout=0.1, activation="relu",
                 normalize_before=False, return_intermediate_dec=False,
                 divide_norm=False):
        super().__init__()
        decoder_layer = TemporalAggregationDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation,
            normalize_before, divide_norm=divide_norm)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TemporalDecoder(
            decoder_layer, num_decoder_layers, decoder_norm,
            return_intermediate=return_intermediate_dec)
        self.history_gate = nn.Linear(d_model, d_model)
        self.d_model = d_model
        self.nhead = nhead
        self.d_feed = dim_feedforward
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat, historical=None):
        # feat: (N, B, C). historical is the fixed accumulated buffer with
        # the same shape, initialized as zeros at sequence start.
        if historical is None:
            historical = torch.zeros_like(feat)
        elif historical.shape != feat.shape:
            historical = torch.zeros_like(feat)
        else:
            historical = historical.to(device=feat.device, dtype=feat.dtype)

        decoded = self.decoder(feat, historical)
        gate = torch.sigmoid(self.history_gate(decoded))
        updated_history = gate * decoded + (1.0 - gate) * historical
        return decoded, updated_history


class TemporalDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, current,
                historical,
                current_key_padding_mask: Optional[Tensor] = None,
                historical_key_padding_mask: Optional[Tensor] = None):
        output = current
        intermediate = []
        for layer in self.layers:
            output = layer(
                output, historical,
                current_key_padding_mask=current_key_padding_mask,
                historical_key_padding_mask=historical_key_padding_mask)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return output


class TemporalAggregationDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, divide_norm=False):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_current = nn.LayerNorm(d_model)
        self.norm_history = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.divide_norm = divide_norm

    def forward_post(self, current, historical,
                     current_key_padding_mask: Optional[Tensor] = None,
                     historical_key_padding_mask: Optional[Tensor] = None):
        current_norm = self.norm_current(current)
        historical_norm = self.norm_history(historical)
        current2 = self.cross_attn(
            query=current_norm,
            key=historical_norm,
            value=historical_norm,
            key_padding_mask=historical_key_padding_mask)[0]
        current = self.norm1(current + self.dropout1(current2))

        current2 = self.self_attn(
            query=current,
            key=self.norm_current(current),
            value=self.norm_current(current),
            key_padding_mask=current_key_padding_mask)[0]
        current = self.norm2(current + self.dropout2(current2))

        current2 = self.linear2(self.dropout(self.activation(self.linear1(current))))
        current = self.norm3(current + self.dropout3(current2))
        return current

    def forward_pre(self, current, historical,
                    current_key_padding_mask: Optional[Tensor] = None,
                    historical_key_padding_mask: Optional[Tensor] = None):
        current_norm = self.norm1(current)
        historical_norm = self.norm_history(historical)
        current = current + self.dropout1(self.cross_attn(
            query=current_norm,
            key=historical_norm,
            value=historical_norm,
            key_padding_mask=historical_key_padding_mask)[0])

        current_norm = self.norm2(current)
        current = current + self.dropout2(self.self_attn(
            query=current_norm,
            key=current_norm,
            value=current_norm,
            key_padding_mask=current_key_padding_mask)[0])

        current_norm = self.norm3(current)
        current = current + self.dropout3(
            self.linear2(self.dropout(self.activation(self.linear1(current_norm)))))
        return current

    def forward(self, current, historical,
                current_key_padding_mask: Optional[Tensor] = None,
                historical_key_padding_mask: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(
                current, historical, current_key_padding_mask,
                historical_key_padding_mask)
        return self.forward_post(
            current, historical, current_key_padding_mask,
            historical_key_padding_mask)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def build_transformer_dec(cfg, hidden_dim):
    return TemporalAggregationDecoder(
        d_model=hidden_dim,
        dropout=cfg.MODEL.TRANSFORMER_DEC.DROPOUT,
        nhead=cfg.MODEL.TRANSFORMER_DEC.NHEADS,
        dim_feedforward=hidden_dim,
        num_decoder_layers=cfg.MODEL.TRANSFORMER_DEC.DEC_LAYERS,
        normalize_before=cfg.MODEL.TRANSFORMER_DEC.PRE_NORM,
        return_intermediate_dec=False,
        divide_norm=cfg.MODEL.TRANSFORMER_DEC.DIVIDE_NORM
    )


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError("activation should be relu/gelu/glu, not {}".format(activation))
