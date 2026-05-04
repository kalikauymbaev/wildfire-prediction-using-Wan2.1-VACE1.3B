import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig

class WildfireWanModel(nn.Module):
    def __init__(self, backbone, stage=1):
        super().__init__()
        self.stage = stage
        
        if isinstance(backbone, dict):
            print(f"--- Config Detected: Building {backbone.get('backbone_name', 'Wan2.1')} ---")
            r = backbone.get('lora_rank', 16)
            alpha = backbone.get('lora_alpha', 32)
            self.hidden_dim = backbone.get('hidden_dim', 512)
            
            # 1. Instantiate the model. 
            # If using a placeholder, wrap it in a Sequential/ModuleDict to give it a name.
            base_model = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
            # In nn.Sequential, the first layer is named '0'
            
            # 2. Update target_modules
            # For Wan2.1 (Transformer), common targets are ["q", "v", "proj", "linear"]
            # For our current placeholder Sequential, we target "0"
            target_modules = ["0", "q", "v", "proj", "linear", "to_q", "to_v"]
            
            peft_config = LoraConfig(
                r=r,
                lora_alpha=alpha,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                modules_to_save=None
            )
            
            try:
                self.backbone = get_peft_model(base_model, peft_config)
            except ValueError:
                # Fallback: If specific names fail, target all Linear layers
                print("Specific targets failed, attempting to target all Linear layers...")
                peft_config.target_modules = None # Some PEFT versions find them automatically
                self.backbone = get_peft_model(base_model, peft_config)
        else:
            self.backbone = backbone
            self.hidden_dim = 512

        # 3. Projection and Heads
        self.input_projection = nn.Linear(327680, self.hidden_dim) 
        self.ir_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 327680),
            nn.Sigmoid()
        )
        self.mask_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 327680),
            nn.Sigmoid()
        )
        self.weather_proj = nn.Sequential(
            nn.Linear(4, 128), # 4 = temp, hum, wind_sin, wind_cos
            nn.ReLU(),
            nn.Linear(128, self.hidden_dim)
        )

    def forward(self, x, weather):
        batch_size = x.size(0)
        x_projected = self.input_projection(x.view(batch_size, -1)) 
        vis_latent = self.backbone(x_projected)
    
        # Handle transformer outputs
        if hasattr(vis_latent, 'last_hidden_state'): vis_latent = vis_latent.last_hidden_state
        elif isinstance(vis_latent, (list, tuple)): vis_latent = vis_latent[0]

        # STAGE 2 FIX: Fuse weather physics with visual features
        weather_latent = self.weather_proj(weather)
        combined_latent = vis_latent + weather_latent # Multimodal Fusion

        pred_ir = self.ir_head(combined_latent).view(batch_size, 5, 1, 256, 256)
        pred_mask = self.mask_head(combined_latent).view(batch_size, 5, 1, 256, 256)
        return pred_ir, pred_mask
