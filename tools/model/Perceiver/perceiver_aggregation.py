# ==================== 导入必要的包 ==================== #
# ----- 系统操作相关的包 ----- #
import time
import sys
import os

# ----- 数据处理相关的包 ----- #
import numpy as np

# ----- 创建模型相关的包 ----- #
import torch
from torch import dropout, nn, einsum
import torch.nn.functional as F

# ----- 获取临时变量 ----- #
# from visualizer import get_local

# ----- 方便创建模型的包 ----- #
from einops import rearrange, repeat
from einops_exts import rearrange_many, repeat_many

# ----- 导入自定义的包 ----- #
from tools.model.Perceiver.perceiver_pos_encoding import build_position_encoding
from tools.model.Perceiver.perceiver_attention import *
# from models.model_set_based_embedding.norm_loss_util import variance_loss


# ==================== 设置常量参数 ==================== #


# ==================== 函数实现 ==================== #

class AggregationBlock(nn.Module):
    def __init__(
            self,
            *,
            depth,
            input_channels=3,
            input_axis=2,
            num_latents=512,
            latent_dim=512,
            num_classes=1000,
            attn_dropout=0.,
            ff_dropout=0.,
            weight_tie_layers=False,
            pos_enc_type='none',
            pre_norm=True,
            post_norm=True,
            activation='geglu',
            last_ln=False,
            ff_mult=4,
            more_dropout=False,
            xavier_init=False,
            query_fixed=False,
            query_xavier_init=False,
            query_type='learned',
            encoder_isab=False,
            first_order=False
    ):
        """
        Args:
            depth: Depth of net.
            input_channels: Number of channels for each token of the input.
            input_axis: Number of axes for input data (2 for images, 3 for video)
            num_latents: Number of element slots
            latent_dim: slot dimension.
            num_classes: Output number of classes.
            attn_dropout: Attention dropout
            ff_dropout: Feedforward dropout
            weight_tie_layers: Whether to weight tie layers (optional).
        """
        super().__init__()
        self.input_axis = input_axis
        self.num_classes = num_classes

        input_dim = input_channels
        self.input_dim = input_channels
        self.pos_enc = build_position_encoding(input_dim, pos_enc_type, self.input_axis)

        self.num_latents = num_latents
        self.query_type = query_type
        self.latent_dim = latent_dim
        self.encoder_isab = encoder_isab
        self.first_order = first_order

        if self.query_type == 'learned':
            self.latents = nn.Parameter(torch.randn(self.num_latents, latent_dim))
            if query_fixed:
                self.latents.requires_grad = False
            if query_xavier_init:
                nn.init.xavier_normal_(self.latents)
        elif self.query_type == 'slot':
            self.slots_mu = nn.init.xavier_uniform_(nn.Parameter(torch.randn(1, 1, latent_dim)),
                                                    gain=nn.init.calculate_gain("linear"))
            self.slots_log_sigma = nn.init.xavier_uniform_(nn.Parameter(torch.randn(1, 1, latent_dim)),
                                                           gain=nn.init.calculate_gain("linear"))
        else:
            raise NotImplementedError

        assert (pre_norm or post_norm)
        self.prenorm = PreNorm if pre_norm else lambda dim, fn, context_dim=None: fn
        self.postnorm = PostNorm if post_norm else nn.Identity
        ff = FeedForward

        # * decoder cross attention layers
        get_cross_attn = \
            lambda: self.prenorm(
                latent_dim,
                Attention(
                    latent_dim, input_dim,
                    heads=4, dim_head=512, dropout=attn_dropout, more_dropout=more_dropout, xavier_init=xavier_init
                ),
                context_dim=input_dim)
        get_cross_ff = lambda: self.prenorm(latent_dim,
                                            ff(latent_dim, dropout=ff_dropout, activation=activation, mult=ff_mult,
                                               more_dropout=more_dropout, xavier_init=xavier_init))
        get_cross_postnorm = lambda: self.postnorm(latent_dim)

        get_cross_attn, get_cross_ff = map(cache_fn, (get_cross_attn, get_cross_ff))

        self.layers = nn.ModuleList([])

        for i in range(depth):
            should_cache = i >= 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}
            # cross attention layer, DETR decoder
            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_postnorm(),
                get_cross_ff(**cache_args),
                get_cross_postnorm()
            ]))

        # Last FC layer
        assert latent_dim == self.num_classes
        self.last_layer = nn.Sequential(
            nn.LayerNorm(latent_dim) if last_ln and not post_norm else nn.Identity()
        )

        self.encoder_output_holder = nn.Identity()
        self.decoder_output_holder = nn.Identity()

    def get_queries(self, b):
        if self.query_type == 'learned':
            ret = repeat(self.latents, 'n d -> b n d', b=b)
        elif self.query_type == 'slot':
            slots_init = torch.randn((b, self.num_latents, self.latent_dim)).cuda()
            ret = self.slots_mu + self.slots_log_sigma.exp() * slots_init
        return ret

    def forward(self, data, mask=None):
        b, *axis, _, device = *data.shape, data.device
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        # concat to channels of data and flatten axis
        pos = self.pos_enc(data)

        data = rearrange(data, 'b ... d -> b (...) d')

        x = self.get_queries(b).type_as(data)

        for i, (cross_attn, pn1, cross_ff, pn2) in enumerate(self.layers):
            if i == len(self.layers) - 1 and self.first_order:
                x = x.detach()
            x = cross_attn(x, context=data, mask=mask, k_pos=pos, q_pos=None) + x
            x = pn1(x)
            x = cross_ff(x) + x
            x = pn2(x)

        x = self.decoder_output_holder(x)
        return self.last_layer(x)


def FeedForward_flamingo(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False)
    )


def FeedForward_flamingo_dropout(dim, mult=4, dropout=0.1):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False)
    )


def FeedForward_flamingo_dropout_more(dim, mult=4, dropout=0.1):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False),
        nn.Dropout(dropout)
    )


class PerceiverAttention_modified(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dim_head=64,
            heads=8,
            xavier_init=False,
            variance_constant=1,
            variance_after_softmax_flag=False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.variance_constant = variance_constant
        self.variance_after_softmax_flag = variance_after_softmax_flag

        if xavier_init:
            self._reset_parameter()

    def _reset_parameter(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_kv.weight)

   #  @get_local('sim')
    def forward(self, x, latents, mask=None, softmax_mode='default', key_mode='default'):
        """
        einstein notation
        b - batch
        t - time
        n - sequence
        d - dimension
        """

        b = x.shape[0]
        h = self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        if key_mode == "concat":
            # print("\t进行 key 和 query 的 concat")
            kv_input = torch.cat((x, latents), dim=-2)
        elif key_mode == "default":
            # print("\t原始的 Attention 过程")
            kv_input = x
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=h)

        q = q * self.scale

        # attention
        sim = einsum('... i d, ... j d  -> ... i j', q, k)

        if exists(mask):
            # print("\t进行 mask")
            # 置 1 进行 mask #
            mask_std = mask.detach().clone()
            mask_attn = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask_attn = repeat(mask_attn, 'b j -> b h () j', h=h)
            sim.masked_fill_(mask_attn, max_neg_value)

            # 将 std 赋予最大值 #
            mask_std = repeat(mask_std, 'b j -> b h j', h=h)

        sim_new = sim - sim.amax(dim=-1, keepdim=True).detach()

        if not self.variance_after_softmax_flag:
            # 计算 variance loss #
            std_parameter = torch.sqrt(sim_new.var(dim=-2) + 0.0001)

            if exists(mask):
                std_parameter = std_parameter.masked_fill(mask_std, self.variance_constant)

            # sim_mean = sim.mean(dim=1)
            # sim_mean_std = torch.sqrt(sim_mean.var(dim=-2) + 0.0001)

            loss_variance = torch.mean(F.relu(self.variance_constant - std_parameter))
            # print("loss_variance: ", loss_variance)

        if softmax_mode == "default":
            # print("\t进行 default Attention")
            attn = sim_new.softmax(dim=-1)
        elif softmax_mode == "slot":
            # print("\t进行 Slot Attention")
            # attention, what we cannot get enough of
            attn = sim_new.softmax(dim=-2)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-7)

        # ----- 提出 nan 值 ----- #
        if torch.isnan(attn).any():
            # print("Nan!")
            import pdb;
            pdb.set_trace()

        if self.variance_after_softmax_flag:
            # 计算 variance loss #
            std_parameter = torch.sqrt(attn.var(dim=-2) + 0.0001)

            if exists(mask):
                std_parameter = std_parameter.masked_fill(mask_std, self.variance_constant)

            # sim_mean = sim.mean(dim=1)
            # sim_mean_std = torch.sqrt(sim_mean.var(dim=-2) + 0.0001)

            loss_variance = torch.mean(F.relu(self.variance_constant - std_parameter))
            # print("loss_variance: ", loss_variance)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)

        out = self.to_out(out)

        return out, loss_variance, attn


class PerceiverResampler(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            num_latents=64,
            ff_mult=4,
            variance_constant=1,
            dropout=0,
            more_drop_flag=False,
            variance_after_softmax_flag=False,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        self.layers = nn.ModuleList([])
        if dropout:
            print("\t使用 Dropout 为: ", dropout)

            if more_drop_flag:
                print("\t使用 more Dropout Flag !")
                for _ in range(depth):
                    self.layers.append(nn.ModuleList([
                        PerceiverAttention_modified(dim=dim, dim_head=dim_head, heads=heads,
                                                    variance_constant=variance_constant,
                                                    variance_after_softmax_flag=variance_after_softmax_flag),
                        nn.LayerNorm(dim),
                        FeedForward_flamingo_dropout_more(dim=dim, mult=ff_mult, dropout=dropout),  # different
                        nn.LayerNorm(dim)
                    ]))
            else:
                for _ in range(depth):
                    self.layers.append(nn.ModuleList([
                        PerceiverAttention_modified(dim=dim, dim_head=dim_head, heads=heads,
                                                    variance_constant=variance_constant,
                                                    variance_after_softmax_flag=variance_after_softmax_flag),
                        nn.LayerNorm(dim),
                        FeedForward_flamingo_dropout(dim=dim, mult=ff_mult, dropout=dropout),
                        nn.LayerNorm(dim)
                    ]))

        else:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PerceiverAttention_modified(dim=dim, dim_head=dim_head, heads=heads,
                                                variance_constant=variance_constant,
                                                variance_after_softmax_flag=variance_after_softmax_flag),
                    nn.LayerNorm(dim),
                    FeedForward_flamingo(dim=dim, mult=ff_mult),
                    nn.LayerNorm(dim)
                ]))

        self.depth = depth

        print(f"\tvariance_constant: {variance_constant}")

    def forward(self, x, mask=None, softmax_mode='default', key_mode='default', variance_flag=False, retrn_attn_flag=False):

        latents = repeat(self.latents, 'n d -> b n d', b=x.shape[0])

        # --- transfer to float16
        latents = latents.half()

        loss_variance = 0

        for attn, ln1, ff, ln2 in self.layers:
            # Attention #
            latents_attn, loss_variance_layer, attn = attn(x, latents, mask=mask, softmax_mode=softmax_mode,
                                                     key_mode=key_mode)
            latents = latents_attn + latents
            latents = ln1(latents)  # 为什么这里的layernormlyzation的weights又变成了float32？？？

            # Feed Forward #
            latents = ff(latents) + latents
            latents = ln2(latents)

            loss_variance += loss_variance_layer

        loss_variance /= self.depth

        if variance_flag:
            return latents, loss_variance
        else:
            if retrn_attn_flag:
                return latents, attn
            return latents