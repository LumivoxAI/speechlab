import math
from typing import Union

import torch
from torch import Tensor, nn


def rtt_half(x: Tensor) -> Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=x1.ndim - 1)


def apply_rotary_pos_emb(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, offset: int = 0
) -> tuple[Tensor, Tensor]:
    """
    Applies Rotary Position Embeddings to query and key tensors.
    """
    cos, sin = (
        cos[offset : q.shape[0] + offset, ...],
        sin[offset : q.shape[0] + offset, ...],
    )
    return (q * cos) + (rtt_half(q) * sin), (k * cos) + (rtt_half(k) * sin)


class StridingSubsampling(nn.Module):
    """
    Strided Subsampling layer used to reduce the sequence length.
    """

    def __init__(
        self,
        subsampling_factor: int = 4,
        feat_in: int = 64,
        feat_out: int = 768,
        conv_channels: int = 768,
    ) -> None:
        super().__init__()
        self._sampling_num = int(math.log(subsampling_factor, 2))
        self._stride = 2
        self._kernel_size = 3
        self._padding = (self._kernel_size - 1) // 2

        layers: list[nn.Module] = []
        in_channels = 1
        for _ in range(self._sampling_num):
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=conv_channels,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    padding=self._padding,
                )
            )
            layers.append(nn.ReLU())
            in_channels = conv_channels

        out_length = self.calc_output_length(feat_in)
        self.out = nn.Linear(conv_channels * out_length, feat_out)
        self.conv = nn.Sequential(*layers)

    def calc_output_length(self, length: int) -> int:
        add_pad = 2 * self._padding - self._kernel_size
        for _ in range(self._sampling_num):
            length = int((length + add_pad) / self._stride) + 1
            length = math.floor(length)
        return int(length)

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(0, 1).unsqueeze(0).unsqueeze(0)
        x = self.conv(x)
        t = x.size(dim=2)
        x = x.transpose(1, 2).reshape(1, t, -1)
        return self.out(x)


class RotaryPositionMultiHeadAttention(nn.Module):
    """
    Rotary Position Multi-Head Attention module.
    """

    def __init__(self, n_head: int, n_feat: int) -> None:
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)

    def forward_qkv(
        self, query: Tensor, key: Tensor, value: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Projects the inputs into queries, keys, and values for multi-head attention.
        """
        b = query.size(0)
        q = self.linear_q(query).view(b, -1, self.h, self.d_k)
        k = self.linear_k(key).view(b, -1, self.h, self.d_k)
        v = self.linear_v(value).view(b, -1, self.h, self.d_k)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def forward_attention(self, value: Tensor, scores: Tensor) -> Tensor:
        """
        Computes the scaled dot-product attention given the projected values and scores.
        """
        b = value.size(0)
        attn = torch.softmax(scores, dim=-1)
        x = torch.matmul(attn, value)
        x = x.transpose(1, 2).reshape(b, -1, self.h * self.d_k)
        return self.linear_out(x)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_emb: list[Tensor],
    ) -> Tensor:
        b, t, _ = value.size()
        query = query.transpose(0, 1).view(t, b, self.h, self.d_k)
        key = key.transpose(0, 1).view(t, b, self.h, self.d_k)
        value = value.transpose(0, 1).view(t, b, self.h, self.d_k)

        cos, sin = pos_emb
        query, key = apply_rotary_pos_emb(query, key, cos, sin, offset=0)

        q, k, v = self.forward_qkv(
            query.view(t, b, self.h * self.d_k).transpose(0, 1),
            key.view(t, b, self.h * self.d_k).transpose(0, 1),
            value.view(t, b, self.h * self.d_k).transpose(0, 1),
        )

        scores = torch.matmul(q, k.transpose(-2, -1) / math.sqrt(self.d_k))
        out = self.forward_attention(v, scores)

        return out


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding module.
    """

    def __init__(self, dim: int, base: int) -> None:
        super().__init__()
        self.dim = dim
        self.base = base

    def create_pe(self, length: int, device: torch.device) -> Tensor | None:
        """
        Creates or extends the rotary positional encoding matrix.
        """
        if hasattr(self, "pe") and self.pe.size(0) >= 2 * length:
            return None
        positions = torch.arange(0, length, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        t = torch.arange(length, device=positions.device).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(positions.device)
        return torch.cat([emb.cos()[:, None, None, :], emb.sin()[:, None, None, :]])

    def extend_pe(self, length: int, device: torch.device) -> None:
        """
        Extends the positional encoding buffer to process longer sequences.
        """
        pe = self.create_pe(length, device)
        if pe is None:
            return
        if hasattr(self, "pe"):
            self.pe = pe
        else:
            self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[Tensor, list[Tensor]]:
        cos_emb = self.pe[0 : x.shape[1]]
        half_pe = self.pe.shape[0] // 2
        sin_emb = self.pe[half_pe : half_pe + x.shape[1]]
        return x, [cos_emb, sin_emb]


class ConformerConvolution(nn.Module):
    """
    Conformer Convolution module.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
            bias=True,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)


class ConformerFeedForward(nn.Module):
    """
    Conformer Feed Forward module.
    """

    def __init__(self, d_model: int, d_ff: int, use_bias=True) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(d_ff, d_model, bias=use_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.activation(self.linear1(x)))


class ConformerLayer(nn.Module):
    """
    Conformer Layer module.
    This module combines several submodules including feed forward networks,
    depthwise separable convolution, and multi-head self-attention
    to form a single Conformer block.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int = 16,
        conv_kernel_size: int = 31,
    ) -> None:
        super().__init__()
        self.fc_factor = 0.5
        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(d_model=d_model, d_ff=d_ff)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConformerConvolution(
            d_model=d_model,
            kernel_size=conv_kernel_size,
        )
        self.norm_self_att = nn.LayerNorm(d_model)
        self.self_attn: nn.Module = RotaryPositionMultiHeadAttention(
            n_head=n_heads,
            n_feat=d_model,
        )

        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model=d_model, d_ff=d_ff)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        pos_emb: Union[Tensor, list[Tensor]],
    ) -> Tensor:
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + x * self.fc_factor

        x = self.norm_self_att(residual)
        x = self.self_attn(x, x, x, pos_emb)
        residual = residual + x

        x = self.norm_conv(residual)
        x = self.conv(x)
        residual = residual + x

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + x * self.fc_factor

        x = self.norm_out(residual)
        return x


class ConformerEncoder(nn.Module):
    """
    Conformer Encoder module.
    This module encapsulates the entire Conformer encoder architecture,
    consisting of a StridingSubsampling layer, positional embeddings, and
    a stack of Conformer Layers.
    It serves as the main component responsible for processing speech features.
    """

    def __init__(
        self,
        feat_in: int = 64,
        n_layers: int = 16,
        d_model: int = 768,
        subsampling_factor: int = 4,
        ff_expansion_factor: int = 4,
        n_heads: int = 16,
        pos_emb_max_len: int = 5000,
        conv_kernel_size: int = 31,
    ) -> None:
        super().__init__()
        self.feat_in = feat_in

        self.pre_encode = StridingSubsampling(
            subsampling_factor=subsampling_factor,
            feat_in=feat_in,
            feat_out=d_model,
            conv_channels=d_model,
        )

        self.pos_enc: nn.Module = RotaryPositionalEmbedding(d_model // n_heads, pos_emb_max_len)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = ConformerLayer(
                d_model=d_model,
                d_ff=d_model * ff_expansion_factor,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
            )
            self.layers.append(layer)

        self.pos_enc.extend_pe(pos_emb_max_len, next(self.parameters()).device)
        self.joint_enc = None  # RNNTJoint.enc

    def input_example(
        self,
        seqlen: int = 200,
    ) -> tuple[Tensor, Tensor]:
        device = next(self.parameters()).device
        return torch.zeros(self.feat_in, seqlen).float().to(device)

    def input_names(self) -> list[str]:
        return ["audio_signal"]

    def output_names(self) -> list[str]:
        return ["encoded_proj"]

    def dynamic_axes(self) -> dict[str, dict[int, str]]:
        return {
            "audio_signal": {1: "seq_len"},
            "encoded_proj": {0: "seq_len"},
        }

    def forward(self, audio_signal: Tensor) -> Tensor:
        audio_signal = torch.log(audio_signal.clamp_(1e-9, 1e9))
        audio_signal = self.pre_encode(audio_signal)
        audio_signal, pos_emb = self.pos_enc(x=audio_signal)

        for layer in self.layers:
            audio_signal = layer(audio_signal, pos_emb)

        return self.joint_enc(audio_signal[0])
