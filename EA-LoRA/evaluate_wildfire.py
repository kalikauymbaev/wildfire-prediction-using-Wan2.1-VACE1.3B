import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc, f1_score, jaccard_score

# 1. Path Setup
REPO_ROOT = "/workspace/Wan2.1/DiffSynth-Studio"
sys.path.append(REPO_ROOT)

from diffsynth.models.model_loader import ModelPool
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.utils.data import VideoData, save_video

def run_physics_evaluation():
    pool = ModelPool()
    device = "cuda"
    
    # 1. Define Paths
    t2v_root = "/workspace/Wan2.1/DiffSynth-Studio/examples/wanvideo/model_training/models/Wan-AI/Wan2.1-T2V-1.3B"
    vace_root = "/workspace/Wan2.1/DiffSynth-Studio/examples/wanvideo/model_training/models/Wan-AI/Wan2.1-VACE-1.3B"
    
    vae_file = os.path.join(t2v_root, "Wan2.1_VAE.pth")
    t2v_dit_file = os.path.join(t2v_root, "diffusion_pytorch_model.safetensors")
    text_enc_file = os.path.join(t2v_root, "models_t5_umt5-xxl-enc-bf16.pth")
    vace_adapter_file = os.path.join(vace_root, "diffusion_pytorch_model.safetensors")
    lora_path = "/workspace/Wan2.1/DiffSynth-Studio/examples/wanvideo/model_training/output/wildfire_ea_lora/epoch-49.safetensors"

    # 2. Load all 4 components into the pool
    # Order matters: Loading the T2V DiT last often makes it the default 'index 0'
    print("Loading VAE...")
    pool.auto_load_model(vae_file)
    print("Loading Text Encoder...")
    pool.auto_load_model(text_enc_file)
    print("Loading VACE Adapter...")
    pool.auto_load_model(vace_adapter_file)
    print("Loading DiT...")
    pool.auto_load_model(t2v_dit_file)

    # 3. Assemble Pipeline
    print("Assembling WanVideoPipeline...")
    pipe = WanVideoPipeline(device=device, torch_dtype=torch.bfloat16)
    
    # Use fetch_model to assign components. 
    pipe.vae = pool.fetch_model("wan_video_vae")
    pipe.text_encoder = pool.fetch_model("wan_video_text_encoder")
    
    # Robust DiT Selection: Use the one from the T2V folder
    # We fetch specifically by path if multiple are loaded to avoid IndexError
    dit_models = pool.fetch_model("wan_video_dit")
    if isinstance(dit_models, list):
        # Find the DiT that belongs to the T2V model set
        pipe.dit = next((m for m in dit_models if "T2V-1.3B" in str(m.device_config)), dit_models[0])
    else:
        pipe.dit = dit_models

    # Setup VACE physics adapter
    vace_model = pool.fetch_model("wan_video_vace")
    if vace_model is not None:
        if os.path.exists(lora_path):
            from safetensors.torch import load_file
            print(f"Applying physics weights to VACE: {lora_path}")
            vace_model.load_state_dict(load_file(lora_path), strict=False)
        pipe.vace = vace_model

    # 4. Tokenizer
    from diffsynth.models.wan_video_text_encoder import HuggingfaceTokenizer
    pipe.tokenizer = HuggingfaceTokenizer(os.path.join(t2v_root, "google/umt5-xxl"))

    # Move to GPU
    pipe.to(device)
    pipe.device = torch.device(device)

    # 5. Evaluation Loop
    dataset_root = "/workspace/Wan2.1/FireSentry-Benchmark-Dataset"
    csv_path = os.path.join(dataset_root, "region_5_validation.csv")
    df = pd.read_csv(csv_path)
    df_r5 = df[df['region'].str.contains('Region D', na=False)] 
    output_dir = "eval_results_region_d"
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, (index, row) in enumerate(tqdm(df_r5.iterrows(), total=len(df_r5), desc="Evaluating Physics")):
        ir_path = os.path.join(dataset_root, row['ir_video_path'])
        mask_path = os.path.join(dataset_root, row['mask_video_path'])

        if not os.path.exists(ir_path):
            continue

        ir_vid = VideoData(ir_path, height=480, width=832)
        num_frames = len(ir_vid)

        with torch.no_grad():
            pred = pipe(
                prompt=row['prompt_ir'], 
                input_video=ir_vid,
                num_frames=num_frames, 
                denoising_strength=1.0, 
                num_inference_steps=50,
                seed=42
            )
        
        # SAFE FILENAME: Uses row 'id' if exists, otherwise uses index 'i'
        sample_id = row.get('id', i)
        video_save_path = os.path.join(output_dir, f"pred_{sample_id}.mp4")
        
        # Save the generated video to inspect visually
        save_video(pred, video_save_path, fps=8) 
        
        # Metrics processing
        pred_np = np.array([np.array(f) for f in pred])
        gt_mask = np.array([np.array(f) for f in VideoData(mask_path, height=480, width=832)])
        
        results.append(compute_metrics(gt_mask, pred_np))

    print("\n" + "="*30)
    print("FINAL REGION 5 PHYSICS RESULTS (REGION D)")
    print("="*30)
    summary = pd.DataFrame(results).mean()
    print(summary)

def compute_metrics(gt, pred):
    # Standardize shapes to (Frames, H, W)
    gt_bin = (gt[..., 0] / 255.0 > 0.5).astype(int).flatten()
    pr_val = (pred[..., 0] / 255.0).flatten()
    pr_bin = (pr_val > 0.5).astype(int)
    
    precision, recall, _ = precision_recall_curve(gt_bin, pr_val)
    return {
        "IoU": jaccard_score(gt_bin, pr_bin, zero_division=0),
        "AUPRC": auc(recall, precision),
        "F1": f1_score(gt_bin, pr_bin, zero_division=0)
    }

if __name__ == "__main__":
    run_physics_evaluation()
