from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.function import ratio2weight


class CEL_Sigmoid(nn.Module):

    def __init__(self, sample_weight=None, size_average=True):
        super(CEL_Sigmoid, self).__init__()

        self.sample_weight = sample_weight
        self.size_average = size_average

    def forward(self, logits, targets):
        batch_size = logits.shape[0]

        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

#         targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))
        
        targets_mask_zero = torch.where( targets.detach().cpu() <= 0.5, torch.zeros(1), targets.detach().cpu() )
        targets_mask_one = torch.where( (targets_mask_zero.detach().cpu() > 0.5) & (targets_mask_zero.detach().cpu() <= 1), torch.ones(1), targets_mask_zero.detach().cpu() )
        targets_mask = torch.where( targets.detach().cpu() >1, torch.ones(1)*2, targets_mask_one.detach().cpu() )
        
        # targets_mask = torch.where( (targets.detach().cpu() > 0.5) & (targets.detach().cpu() <= 1), torch.ones(1), torch.ones(1)*2 )
        if self.sample_weight is not None:
            weight = ratio2weight(targets_mask, self.sample_weight)
            if torch.cuda.is_available():
                loss = (loss * weight.cuda())
            else:
                loss = (loss * weight.cpu())

        loss = loss.sum() / batch_size if self.size_average else loss.sum()

        return loss
