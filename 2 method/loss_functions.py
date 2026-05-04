import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips

class WildfireLoss(nn.Module):
    def __init__(self, fire_weight=50.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss() # NEW: Added for tracking
        self.perceptual = lpips.LPIPS(net='vgg')
        self.fire_weight = fire_weight

    def dice_loss(self, pred, target):
        smooth = 1e-6
        iflat, tflat = pred.contiguous().view(-1), target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

    def forward(self, pred_ir, target_ir, pred_mask, target_mask):
        # 1. IR Reconstruction
        recon_l1 = self.l1(pred_ir, target_ir)
        recon_mse = self.mse(pred_ir, target_ir) # MSE Metric
        
        # 2. Perceptual Loss
        b, t, c, h, w = pred_ir.shape
        p_loss = self.perceptual(pred_ir.view(-1, c, h, w), target_ir.view(-1, c, h, w)).mean()

        # 3. Weighted Mask Loss (Class Imbalance Fix)
        weights = target_mask * self.fire_weight + (1 - target_mask)
        bce_loss = F.binary_cross_entropy(pred_mask, target_mask, weight=weights)
        dice = self.dice_loss(pred_mask, target_mask)
        
        # We use L1 + Perceptual + Weighted BCE + Dice for training
        # MSE is calculated but often kept separate as a benchmark metric
        total_loss = recon_l1 + (0.5 * p_loss) + (2.0 * (bce_loss + dice))
        
        # We return the total_loss and the MSE for logging
        return total_loss, recon_mse
