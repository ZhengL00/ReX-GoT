import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import matplotlib.font_manager as font_manager

def sq_euclidean_dist(x1, x2):
    diff = x1 - x2
    diff_sq = diff * diff
    diff_sum = torch.sum(diff_sq, axis=-1)
    return diff_sum


def fro_norm(x1, x2):
    diff = x1 - x2
    fro_loss = torch.linalg.norm(diff, dim=1).mean()
    return fro_loss


def fro_norm_pos(x1, x2):
    pos_mask = (x2 > 0.0).float()
    diff = x1 - x2
    fro_loss = (torch.linalg.norm(diff, dim=1) * pos_mask).mean()
    return fro_loss


def smooth_l1_loss(
    input: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    beta = 1
    if beta < 1e-5:
        loss = torch.abs(input - target)
    else:
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n**2 / beta, n - 0.5 * beta)


    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, features=None, labels=None, dist_matrix=None):

        if dist_matrix is None:
            dist_matrix = torch.cdist(features, features)

        mask_pos = labels.eq(1).float()
        mask_neg = labels.eq(0).float()

        non_zero_mask = (torch.sum(labels, dim=1) > 0.0).float()

        pos_loss = mask_pos * torch.clamp(self.margin - torch.sqrt(dist_matrix), min=0)

        pos_loss = torch.mean(pos_loss * non_zero_mask[:, None])
        neg_loss = mask_neg * torch.clamp(torch.sqrt(dist_matrix) - self.margin, min=0)
        neg_loss = torch.mean(neg_loss * non_zero_mask[:, None])

        return pos_loss + neg_loss




class SmoothLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(SmoothLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = torch.norm(anchor - positive, dim=1)
        distance_negative = torch.norm(anchor - negative, dim=1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()




def bce_loss(features, target):
    features = F.normalize(features, p=2.0)
    input = torch.matmul(features, features.t())
    max_val = (-input).clamp(min=0)
    loss = (
        input
        - input * target
        + max_val
        + ((-max_val).exp() + (-input - max_val).exp()).log()
    )

    return loss.mean()


def ce_loss(pred, target):

    loss = -(target * (pred + 1e-9).log()).mean()

    return loss



def smooth_loss1(pred, target, smoothing=0.1):

    smooth_target = torch.full_like(pred, smoothing / (pred.size(1) - 1))
    smooth_target.scatter_(1, target, 1 - smoothing)

    loss = -(smooth_target * pred.log()).sum(dim=1).mean()

    return loss


class SupContrastiveLoss(nn.Module):
    def __init__(self, temp):
        super(SupContrastiveLoss, self).__init__()
        self.temperature = temp

    def small_val(self, dtype):
        return torch.finfo(dtype).tiny

    def neg_inf(self, dtype):
        return torch.finfo(dtype).min

    def logsumexp(self, x, keep_mask=None, add_one=True, dim=1):
        if keep_mask is not None:
            x = x.masked_fill(~keep_mask, self.neg_inf(x.dtype))
        if add_one:
            zeros = torch.zeros(
                x.size(dim - 1), dtype=x.dtype, device=x.device
            ).unsqueeze(dim)
            x = torch.cat([x, zeros], dim=dim)

        output = torch.logsumexp(x, dim=dim, keepdim=True)
        if keep_mask is not None:
            output = output.masked_fill(~torch.any(keep_mask, dim=dim, keepdim=True), 0)
        return output

    def forward(self, features=None, labels=None, perm_indexes=None):

        features = F.normalize(features, p=2.0)
        mat = torch.matmul(features, features.t())
        if perm_indexes is not None:
            mat = torch.index_select(
                mat, 0, torch.tensor([perm_indexes]).to(mat.device)
            )
            labels = torch.index_select(
                labels, 0, torch.tensor([perm_indexes]).to(mat.device)
            )

        pos_mask = labels.eq(1).float()
        neg_mask = labels.eq(0).float()
        mat = mat / self.temperature
        mat_max, _ = mat.max(dim=1, keepdim=True)
        mat = mat - mat_max.detach()

        denominator = self.logsumexp(
            mat, keep_mask=(pos_mask + neg_mask).bool(), add_one=False, dim=1
        )
        log_prob = mat - denominator
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (
            pos_mask.sum(dim=1) + self.small_val(mat.dtype)
        )

        return (-mean_log_prob_pos).mean()
