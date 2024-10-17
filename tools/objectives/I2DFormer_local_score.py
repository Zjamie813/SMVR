"""
    本代码用于: 完成 I2DFormer 的细粒度打分函数实现
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

# ----- 导入自定义的包 ----- # 
from models.local_score_function.local_score_util import get_logits, Attention_layer, max_pool, mean_pool

# ==================== 设置常量参数 ==================== #


# ==================== 函数实现 ==================== #
class I2DFormer_logits_local(nn.Module):
    def __init__(self, feature_dim, heads, pooling_method):
        super().__init__()

        # 定义池化方式 # 
        if pooling_method == "max":
            self.pool_func = max_pool 
        elif pooling_method == "mean":
            # print("\t使用 mean !")
            self.pool_func = mean_pool 

        # 定义 local 的 attention layer
        self.attention_layer = Attention_layer(d_model=feature_dim, heads=heads)

        # 定义 local 打分函数 # 
        self.local_linear = nn.Linear(feature_dim, 1)

    def forward(self, image_local_features, text_local_features, key_padding_mask=None):

        # ----- 计算 local attention ----- # 
        attention_values_local = self.attention_layer(image_local_features, text_local_features, key_padding_mask=key_padding_mask)

        # ----- 计算 local loss ----- # 
        values_local_pool = self.pool_func(attention_values_local, dim=2)
        logits_per_image_local = self.local_linear(values_local_pool).squeeze(2)
        
        return logits_per_image_local
