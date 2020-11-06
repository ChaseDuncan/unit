"""
Dice loss 3D
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


def dice_score(preds, targets):
    if len(preds.size()) == 5:
        num = 2*torch.einsum('bcijk, bcijk ->bc', [preds, targets])
        denom = torch.einsum('bcijk, bcijk -> bc', [preds, preds]) +\
            torch.einsum('bcijk, bcijk -> bc', [targets, targets]) + 1e-8
    if len(preds.size()) == 4:
        num = 2*torch.einsum('bcij, bcij ->bc', [preds, targets])
        denom = torch.einsum('bcij, bcij -> bc', [preds, preds]) +\
            torch.einsum('bcij, bcij -> bc', [targets, targets]) + 1e-8

    proportions = torch.div(num, denom) 
    return torch.einsum('bc->c', proportions)


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
    
    def forward(self, mu, logvar, device):
        return -0.5*torch.mean(torch.ones(mu.size()).to(device)+logvar-torch.exp(logvar)-torch.square(mu))
    #def forward(self, mu, logvar, N):
    #    sum_square_mean = torch.einsum('i,i->', mu, mu)
    #    sum_log_var = torch.einsum('i->', logvar)
    #    sum_var = torch.einsum('i->', torch.exp(logvar))
    #    #print(f'ssm: {sum_square_mean}\tslv: {sum_log_var}\t svr: {sum_var}\t') 
    #    return float(1/N)*(sum_square_mean+sum_var-sum_log_var-N)


class VAEDiceLoss(nn.Module):
    def __init__(self, device):
        super(VAEDiceLoss, self).__init__()
        self.avgdice = AvgDiceLoss()
        self.kl = KLLoss()
        self.device = device

    def forward(self, output, targets):
        #ad = 0.8*self.avgdice(output['seg_map'], targets)
        target = targets['target']
        d = -torch.einsum('c->', dice_score(output['seg_map'], target))
        ms = 0.1*F.mse_loss(output['recon'], targets['src'], reduction='sum')
        kl = 0.1*self.kl(output['mu'], output['logvar'], self.device)

        return d + ms + kl
        #return ad + ms + kl
        #return self.avgdice(output['seg_map'], targets)\
        #    + 0.1*F.mse_loss(output['recon'], targets['src'])\
        #    + 0.1*self.kl(output['mu'], output['logvar'], 256)

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.dice = AvgDiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, logits, targets):
        target = targets['target']
        return self.dice(preds, targets) + self.bce(logits[:, 0, :, :, :].squeeze(), target[:, 0, :, :, :].squeeze())\
                + self.bce(logits[:, 1, :, :, :].squeeze(), target[:, 1, :, :, :].squeeze())\
                + self.bce(logits[:, 2, :, :, :].squeeze(), target[:, 2, :, :, :].squeeze())

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        return self.bce(logits[:, 0, :, :, :].squeeze(), target[:, 0, :, :, :].squeeze())\
                + self.bce(logits[:, 1, :, :, :].squeeze(), target[:, 1, :, :, :].squeeze())\
                + self.bce(logits[:, 2, :, :, :].squeeze(), target[:, 2, :, :, :].squeeze())

class AvgDiceLoss(nn.Module):
    def __init__(self):
        super(AvgDiceLoss, self).__init__()
    
    def forward(self, preds, target):
        proportions = dice_score(preds, target)
        avg_dice = torch.einsum('c->', proportions) / (target.shape[0]*target.shape[1])
        return 1 - avg_dice

