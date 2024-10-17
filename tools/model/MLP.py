import torch.nn as nn
import torch

class MLP_layer(nn.Module):
    def __init__(self, d_model_in, d_model_out, d_model_mid, dropout_rate):
        super().__init__()
        self.linear1 = nn.Linear(d_model_in, d_model_mid)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(d_model_mid, d_model_out)
        self.ln = nn.LayerNorm(d_model_out)

        # nn.linear() 默认就是 Kaiming初始化, 效果已经很好啦 !

    def forward(self, x, x_res=None): # 默认是None
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = x.float()
        if x_res is not None:
            x = x + x_res
        x = self.ln(x)
        return x

class MLP_layer_return2(nn.Module):
    def __init__(self, d_model_in, d_model_out, d_model_mid, dropout_rate):
        super().__init__()
        self.linear1 = nn.Linear(d_model_in, d_model_mid)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(d_model_mid, d_model_out)
        self.ln = nn.LayerNorm(d_model_out)

        # nn.linear() 默认就是 Kaiming初始化, 效果已经很好啦 !

    def forward(self, x, x_res=None):  # 默认是None
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = x.float()
        if x_res is not None:
            out = x + x_res
        out = self.ln(out)
        return out, x


class MLP_image_layer_2(nn.Module):
    def __init__(self, d_model_in=768, d_model_out=64, d_model_mid=1024, dropout_rate=0.3, image_mlp_res_flag=False):
        super().__init__()

        # ----- 初始化 res 标志 ----- #
        self.image_mlp_res_flag = image_mlp_res_flag

        if image_mlp_res_flag:
            scale = d_model_in ** -0.5
            self.proj = nn.Parameter(scale * torch.randn(d_model_in, d_model_out))

        # ---------- 第一种写法 ---------- #
        # 默认的 dropout 比例 #
        self.dropout = nn.Dropout(dropout_rate)

        # 输入值的 Layer norm #
        self.ln_0 = nn.LayerNorm(d_model_in)

        # 1st 的 Layer norm #
        self.linear_1 = nn.Linear(d_model_in, d_model_mid)
        self.ln_1 = nn.LayerNorm(d_model_mid)
        self.relu_1 = nn.ReLU(inplace=True)

        self.linear2 = nn.Linear(d_model_mid, d_model_out)
        self.ln_2 = nn.LayerNorm(d_model_out)
        self.relu_2 = nn.ReLU(inplace=True)

    def forward(self, x):

        if self.image_mlp_res_flag:
            x_res = x @ self.proj

        # ---------- 第一种写法 ---------- #
        # x = self.ln_0(x)

        x = self.relu_1(self.linear_1(x))
        x = self.ln_1(x)
        x = self.dropout(x)

        x = self.relu_2(self.linear2(x))

        if self.image_mlp_res_flag:
            x = x + x_res

        x = self.ln_2(x)
        x = self.dropout(x)

        return x
