import torch
import torch.nn as nn

# Mean square error loss: 用来约束特征相似度的
def mse_loss(x, target, mean=True):
    loss_fn = nn.MSELoss()
    loss = loss_fn(x, target)
    if not mean:
        loss_fn = nn.MSELoss(size_average=False)
        loss = loss_fn(x, target)
    return loss

def smoothl1loss(x, target, beta=2.0, mean=True):
    loss_fn = nn.SmoothL1Loss(beta=beta)
    loss = loss_fn(x, target)
    if mean:
        return loss.mean()
    else:
        return loss

