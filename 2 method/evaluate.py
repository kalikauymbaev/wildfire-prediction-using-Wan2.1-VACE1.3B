import sys
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.wan_wrapper import WildfireWanModel
from data.dataset import FireSentryDataset 

def calculate_metrics(pred, target, threshold=0.1):
    """Calculates IoU and F1-Score for binary fire masks."""
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()
    
    tp = (pred_bin * target_bin).sum()
    fp = (pred_bin * (1 - target_bin)).sum()
    fn = ((1 - pred_bin) * target_bin).sum()
    
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return iou.item(), f1.item()

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--checkpoint', type=str, default="checkpoints/stage1_latest.pth")
    parser.add_argument('--data_dir', type=str, default="/workspace/Wan2.1/FireSentry-Benchmark-Dataset")
    parser.add_argument('--recursive_steps', type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = f"./eval_results_stage{args.stage}"
    os.makedirs(output_dir, exist_ok=True)

    config = {
        'model': {
            'backbone_name': 'Wan2.1',
            'lora_rank': 64,
            'lora_alpha': 128,
            'hidden_dim': 512,
            'weather_dim': 4
        }
    }

    model = WildfireWanModel(config).to(device)
    if os.path.exists(args.checkpoint):
        print(f"✅ Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    val_dataset = FireSentryDataset(args.data_dir, regions=['Region D'], augment=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # NEW: Initialize Metric Accumulators
    total_iou, total_f1, total_mse = 0, 0, 0
    mse_criterion = nn.MSELoss()

    print(f"🚀 Evaluating Stage {args.stage} with MSE tracking...")

    with torch.no_grad():
        for i, (past_ir, gt_ir, gt_mask, weather) in enumerate(tqdm(val_loader)):
            past_ir, gt_ir, gt_mask, weather = past_ir.to(device), gt_ir.to(device), gt_mask.to(device), weather.to(device)

            # Inference
            pred_ir, pred_mask = model(past_ir, weather)

            # --- Metrics Calculation ---
            # 1. Mask Metrics (IoU/F1)
            iou, f1 = calculate_metrics(pred_mask, gt_mask)
            
            # 2. IR Metrics (MSE) - Measures reconstruction quality
            mse = mse_criterion(pred_ir, gt_ir).item()

            total_iou += iou
            total_f1 += f1
            total_mse += mse

            if i % 10 == 0:
                # Visualization with both IR and Mask
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                axes[0].imshow(gt_ir[0, -1, 0].cpu(), cmap='magma')
                axes[0].set_title(f"GT IR (MSE: {mse:.4f})")
                axes[1].imshow(pred_mask[0, -1, 0].cpu(), cmap='hot')
                axes[1].set_title(f"Pred Mask (IoU: {iou:.4f})")
                plt.savefig(f"{output_dir}/sample_{i}.png")
                plt.close()

    # Final Report
    count = len(val_loader)
    print("\n" + "="*35)
    print(f"STAGE {args.stage} FINAL METRICS")
    print(f"Avg IoU: {total_iou / count:.4f}")
    print(f"Avg F1:  {total_f1 / count:.4f}")
    print(f"Avg MSE: {total_mse / count:.6f} (IR Recon)") # MSE is usually small
    print("="*35)

if __name__ == "__main__":
    evaluate()
