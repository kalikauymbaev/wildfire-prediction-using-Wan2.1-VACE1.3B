import os
import pandas as pd

base_dir = "./"
regions = ["Region A", "Region B", "Region C", "Region D", "Region E"]
all_data = []

for region in regions:
    region_path = os.path.join(base_dir, region)
    ir_dir = os.path.join(region_path, "Infrared Videos")
    mask_dir = os.path.join(region_path, "Fire Mask Videos")
    env_file = os.path.join(region_path, "Environmental Info", "P171-1020.csv") # Path to env data

    if not os.path.exists(ir_dir): continue

    for filename in os.listdir(ir_dir):
        if filename.endswith(".mp4"):
            # Ensure the mask exists for this IR video
            if os.path.exists(os.path.join(mask_dir, filename)):
                all_data.append({
                    "region": region,
                    "ir_video_path": os.path.join(ir_dir, filename),
                    "mask_video_path": os.path.join(mask_dir, filename),
                    "env_info_path": env_file,
                    "prompt_ir": "Satellite thermal view of wildfire spread dynamics",
                    "prompt_mask": "Dynamic fire front boundary mask"
                })

df = pd.DataFrame(all_data)
df.to_csv("physfire_train.csv", index=False)
print(f"Generated {len(df)} training pairs across {len(regions)} regions.")
