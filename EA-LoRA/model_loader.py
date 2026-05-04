import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import lpips
from sklearn.metrics import precision_recall_curve, auc, f1_score, jaccard_score, mean_squared_error
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from diffsynth.models.model_loader import ModelPool
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

# --- Metric Logic ---

def compute_mask_metrics(gt_mask_video, pred_video):
    """Fire Mask Quality: AUPRC, F1, IoU, MSE"""
    # Using the first channel as a proxy for fire mask intensity
    gt_flat = (gt_mask_video[..., 0] / 255.0).flatten()
    pred_flat = (pred_video[..., 0] / 255.0).flatten()
    
    # AUPRC (Area Under Precision-Recall Curve)
    precision, recall, _ = precision_recall_curve((gt_flat > 0.5).astype(int), pred_flat)
    auprc_val = auc(recall, precision)
    
    # Binarized metrics for F1 and IoU
    gt_bin = (gt_flat > 0.5).astype(int)
    pred_bin = (pred_flat > 0.5).astype(int)
    
    return {
        "AUPRC": auprc_val,
        "F1": f1_score(gt_bin, pred_bin, zero_division=0),
        "IoU": jaccard_score(gt_bin, pred_bin, zero_division=0),
        "MSE": mean_squared_error(gt_flat, pred_flat)
    }

def compute_infra_metrics(gt_video, pred_video, lpips_model):
    """Infrared Quality: PSNR, SSIM, LPIPS"""
    device = "cuda"
    psnr_m = PeakSignalNoiseRatio().to(device)
    ssim_m = StructuralSimilarityIndexMeasure().to(device)

    # Convert to Tensors [B, C, H, W] normalized to [0, 1]
    gt_t = torch.from_numpy(gt_video).permute(0, 3, 1, 2).float().to(device) / 255.0
    pr_t = torch.from_numpy(pred_video).permute(0, 3, 1, 2).float().to(device) / 255.0

    psnr = psnr_m(pr_t, gt_t).item()
    ssim = ssim_m(pr_t, gt_t).item()
    
    # LPIPS expects [-1, 1] range
    lp_val = lpips_model(pr_t * 2 - 1, gt_t * 2 - 1).mean().item()

    return {"PSNR": psnr, "SSIM": ssim, "LPIPS": lp_val}

# --- Main Evaluation ---

def run_evaluation():
    # CONFIGURATION
    csv_path = "/workspace/Wan2.1/FireSentry-Benchmark-Dataset/region_5_validation.csv" # Your CSV file name
    weights_path = "/workspace/Wan2.1/DiffSynth-Studio/examples/wanvideo/model_training/output/wildfire_ea_lora/epoch-49.safetensors"
    device = "cuda"
    
    # 1. Load Data & Filter for Region 5
    df = pd.read_csv(csv_path)
    # Using 'Region A' if your CSV uses letters as seen in the screenshot, or 'Region 5'
    target_region = "Region D" 
    df_r5 = df[df['region'].str.contains(target_region, na=False)]
    
    if df_r5.empty:
        print(f"No samples found for {target_region}. Available regions: {df['region'].unique()}")
        return

    # 2. Initialize Wan2.1 VACE
model_pool = ModelPool(device=device)

    model_pool.load_models(["Wan-AI/Wan2.1-VACE-1.3B"])
    
    # 3. Load your trained weights onto the pipeline
    # Note: verify if your weights are for the DiT or the VAE
    model_pool.load_lora(weights_path)

    pipe = WanVideoPipeline.from_model_manager(model_pool)
    pipe.to(device)
    
    lpips_model = lpips.LPIPS(net='alex').to(device)
    
    all_results = []

    for _, row in tqdm(df_r5.iterrows(), total=len(df_r5), desc=f"Evaluating {target_region}"):
        # Load Video sequences
        # Wan2.1-VACE usually expects 480x832 for 1.3B model
        gt_ir = np.array([np.array(f) for f in VideoData(row['ir_video_path'], height=480, width=832)])
        gt_mask = np.array([np.array(f) for f in VideoData(row['mask_video_path'], height=480, width=832)])
        
        # Inference (Reconstruction)
        # We pass prompt_ir for conditioning
        pred_video = pipe(
            prompt=row['prompt_ir'],
            video_data=VideoData(row['ir_video_path'], height=480, width=832),
            num_inference_steps=50
        )
        
        # 3. Calculate Metrics
        m_results = compute_mask_metrics(gt_mask, pred_video)
        i_results = compute_infra_metrics(gt_ir, pred_video, lpips_model)
        
        all_results.append({**m_results, **i_results})

    # Final Report
    summary = pd.DataFrame(all_results).mean()
    print("\n" + "="*40)
    print(f"RESULTS FOR {target_region}")
    print("="*40)
    print(summary)
    print("="*40)

if __name__ == "__main__":
    run_evaluation()
        

