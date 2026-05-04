import sys
import os
import torch
import argparse
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.wan_wrapper import WildfireWanModel
from models.loss_functions import WildfireLoss
from data.dataset import FireSentryDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/workspace/Wan2.1/FireSentry-Benchmark-Dataset")
    parser.add_argument('--stage', type=int, default=1, help="1: Visual, 2: Physics-Fusion, 3: Simulation")
    parser.add_argument('--holdout', type=str, default="Region D")
    parser.add_argument('--epochs', type=int, default=10) 
    # NEW: Added to allow loading previous stage weights
    parser.add_argument('--load_weights', type=str, default=None)
    args = parser.parse_args()

    # Define the 5 regions based on your dataset structure[cite: 14]
    all_regions = ["Region A", "Region B", "Region C", "Region D", "Region E"]
    train_regions = [r for r in all_regions if r != args.holdout]

    config = {
        'model': {
            'backbone_name': 'Wan2.1-VACE-1.3B',
            'lora_rank': 64,
            'lora_alpha': 128,
            'weather_dim': 4, # temp, humidity, wind_sin, wind_cos[cite: 14]
            'hidden_dim': 512
        }
    }

    print(f"🚀 Initializing Stage {args.stage} training...")
    print(f"Training on: {train_regions}")
    print(f"Holding out: {args.holdout}")

    # Initialize Model with the specific stage[cite: 13]
    model = WildfireWanModel(config, stage=args.stage).cuda()
    
    # NEW: Load weights if moving from Stage 1 -> 2 or 2 -> 3[cite: 10]
    if args.load_weights and os.path.exists(args.load_weights):
        print(f"🔄 Loading weights from {args.load_weights}...")
        model.load_state_dict(torch.load(args.load_weights, weights_only=True), strict=False)
    
    criterion = WildfireLoss().cuda()
    
    # Lower learning rate for Stage 3 to maintain stability
    lr = 1e-4 if args.stage < 3 else 5e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    dataset = FireSentryDataset(args.data_dir, train_regions, augment=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model.train()
    for epoch in range(args.epochs):
        for batch_idx, (past_ir, future_ir, future_mask, weather) in enumerate(loader):
            optimizer.zero_grad()
            
            # --- STAGE 3 LOGIC: Recursive Forward Pass ---
            # If Stage 3, we simulate an extra step to teach the model stability
            if args.stage == 3:
                pred_ir_1, pred_mask_1 = model(past_ir.cuda(), weather.cuda())
                # Feed the prediction back as the next input
                pred_ir, pred_mask = model(pred_ir_1, weather.cuda())
            else:
                # Stage 1 & 2 use standard one-step prediction[cite: 14]
                pred_ir, pred_mask = model(past_ir.cuda(), weather.cuda())
            
            # --- LOSS FIX: Unpack tuple (total_loss, recon_mse) ---
            loss, mse = criterion(pred_ir, future_ir.cuda(), pred_mask, future_mask.cuda())
            
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"E{epoch} B{batch_idx} | Loss: {loss.item():.4f} | MSE: {mse.item():.6f}")
                
                # --- DYNAMIC FILENAME FIX: Save by Stage ---
                latest_path = os.path.join(checkpoint_dir, f"stage{args.stage}_latest.pth")
                torch.save(model.state_dict(), latest_path)

        # Save at the end of every epoch[cite: 14]
        epoch_path = os.path.join(checkpoint_dir, f"stage{args.stage}_epoch_{epoch}.pth")
        torch.save(model.state_dict(), epoch_path)
        print(f"✅ Saved checkpoint: {epoch_path}")

if __name__ == "__main__":
    main()
