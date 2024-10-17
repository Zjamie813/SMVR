"""
    本代码用于: 用于实现 local score 的辅助函数
    创建时间: 2023 年 11 月 12 日
    创建人: MorningStar
    最后一次修改时间: 2023 年 11 月 12 日
"""
# ==================== 导入必要的包 ==================== #
# ----- 系统操作相关的包 ----- #
import time
import sys
import os 

# ----- 数据处理相关的包 ----- #
import numpy as np 
import json 
import math 

# ----- 创建模型相关的包 ----- # 
import torch
from torch import dropout, nn, einsum
import torch.nn.functional as F

# ----- 方便创建模型的包 ----- # 
from einops import rearrange, repeat 
from einops_exts import rearrange_many, repeat_many

# ==================== 设置常量参数 ==================== #


# ==================== 函数实现 ==================== #
def max_pool(x, dim):
    """max pooling"""

    x_max = torch.max(x, dim=dim)[0].contiguous()
    
    return x_max 

def mean_pool(x, dim):
    """average pooling"""

    x_mean = torch.mean(x, dim=dim) 

    return x_mean 

def max_neg_value(dtype):
    """返回 dtype 类型"""
    return -torch.finfo(dtype).max

def get_logits(dense_feat_1, dense_feat_2):
    """完成 Logits 的矩阵乘法"""
    i, j, k = dense_feat_1.shape
    l, m, k = dense_feat_2.shape
    dense_feat_1 = dense_feat_1.reshape(-1, k)
    dense_feat_2 = dense_feat_2.reshape(-1, k)
    final_logits_1 = dense_feat_1 @ dense_feat_2.t()
    final_logits_1 = final_logits_1.reshape(i, j, l, m).permute(0,2,1,3)

    return final_logits_1

class Attention_layer(nn.Module):
    def __init__(self, d_model=64, heads=1):
        super().__init__()

        # ----- 初始化 Attention 的相关参数 ----- #  
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.scale = d_model ** -0.5

        # ----- 创建 linear 函数 ----- # 
        self.W_Q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.W_K = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.W_V = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(self.d_model, self.d_model, bias=False)

        self.norm = nn.LayerNorm(self.d_model)

        # ----- 初始化参数 ----- # 
        attn_std = self.d_model ** -0.5
        nn.init.normal_(self.W_Q.weight, std=attn_std)
        nn.init.normal_(self.W_K.weight, std=attn_std)
        nn.init.normal_(self.W_K.weight, std=attn_std)

    # @get_local('scores')
    def forward(self, query, key, key_padding_mask=None, return_score_flag=False):

        query = rearrange(query, 'i q (h d) -> i h q d', h=self.heads, d=self.head_dim)
        key = rearrange(key, 't k (h d) -> t h k d', h=self.heads, d=self.head_dim)

        # ----- 经过线性映射 ----- # 
        query_new = self.W_Q(query)
        key_new = self.W_K(key)
        value_new = self.W_V(key)

        """应该正确的算法 !"""
        scores =  einsum("i h q d, t h k d -> i t h q k", query_new, key_new) / math.sqrt(self.d_model)
        # print("scores: ", scores.shape)

        if key_padding_mask is not None:
            src_key_padding_mask = repeat(key_padding_mask, 't k -> i h q t k', i=scores.shape[0], h=scores.shape[2], q=scores.shape[3])
            src_key_padding_mask = rearrange(src_key_padding_mask, 'i h q t k -> i t h q k')
            scores.masked_fill_(src_key_padding_mask==False, torch.finfo(scores.dtype).min)
            # print("scores: ", scores.shape)
            # assert (scores[0, :, 0, 0, :][key_padding_mask==False] == scores[1, :, 1, 1, :][key_padding_mask==False]).all()

        # 得到 value 值 # 
        p_attn = F.softmax(scores, dim = -1)
        # print("p_attn example: ", p_attn[0, 0, 0])
        value_attention = einsum('i t h q k, t h k d -> i t h q d', p_attn, value_new)
        value_attention = rearrange(value_attention, 'i t h q d -> i t q (h d)', h=self.heads)

        value_attention = self.fc_out(value_attention)

        value_attention = self.norm(value_attention)

        if return_score_flag:
            return value_attention, scores
        else:
            return value_attention


class Attention_pool_layer(nn.Module):
    """类似于 PerceiverResampler"""
    def __init__(self, d_model=64, heads=1, num_latents=1, dropout=0.1):
        super().__init__()

        # 创建 Attention Layer 
        self.attention_layer = Attention_layer(d_model=d_model, heads=heads)

        # 创建可查询变量
        self.latents = nn.Parameter(torch.randn(num_latents, d_model))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, return_score_flag=False, mean_res_flag=False):
        """x 为要 Pool 的特征, x 的 Shape 大小: [bz, query, dim]"""

        latents = repeat(self.latents, 'n d -> b n d', b = x.shape[0])

        if mean_res_flag:
            x_mean = x.mean(dim=1)

        latents = self.attention_layer(latents, x, key_padding_mask=key_padding_mask)
        latents = self.dropout(latents)

        if mean_res_flag:
            latents = latents + x_mean

        return latents