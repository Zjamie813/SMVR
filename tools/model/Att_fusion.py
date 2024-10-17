import numpy as np
import torch
import torch.nn as nn
from model.clip_model import Transformer, LayerNorm

class MultiHeadSelfAttetion(nn.Module):
    """Self-attention module by Lin, Zhouhan, et al. ICLR 2017"""
    def __init__(self, n_head, d_in, d_hidden):
        super(MultiHeadSelfAttetion, self).__init__()

        self.n_head = n_head
        self.w_1 = nn.Linear(d_in, d_hidden, bias=False)
        self.w_2 = nn.Linear(d_hidden, n_head, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)

    def forward(self, x, mask=None):
        # x : [bs, seqlen, d_feat]
        attn = self.w_2(self.tanh(self.w_1(x)))
        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1).permute(1, 2, 0)
            attn.masked_fill_(mask, -np.inf)  # for the changed length tensor like texts, 为true填充对应的值
        attn = self.softmax(attn)  # 负无穷softmax之后会变为0

        output = torch.bmm(attn.transpose(2, 1), x) # bs, num_head, d_feat
        if output.shape[1] == 1:  # num_head=1
            output = output.squeeze(1)
        return output, attn

class MHSA_fusion_from_irra(nn.Module):
    "存在kqv"
    def __init__(self):
        super(MHSA_fusion_from_irra).__init__()
        self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                self.embed_dim // 64,
                                                batch_first=True)  # param 1: embed_dim, param 2: num_head, embed_dim是总维度=单头注意力维度*头个数
        self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                   layers=args.cmt_depth,
                                                   heads=self.embed_dim //
                                                         64)
        scale = self.cross_modal_transformer.width ** -0.5

        self.ln_pre_t = LayerNorm(self.embed_dim)
        self.ln_pre_i = LayerNorm(self.embed_dim)
        self.ln_post = LayerNorm(self.embed_dim)

        proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
        attn_std = scale
        fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
        for block in self.cross_modal_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # init cross attn
        nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

    def foward(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

