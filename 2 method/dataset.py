import os
import pandas as pd
import torch
import cv2
import numpy as np
import random
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

class FireSentryDataset(Dataset):
    def __init__(self, root_dir, regions, frames_past=5, frames_future=5, img_size=(256, 256), augment=False):
        self.root_dir = root_dir
        self.regions = regions
        self.frames_past = frames_past
        self.frames_future = frames_future
        self.img_size = img_size
        self.augment = augment  # Toggle for training

        csv_path = os.path.join(root_dir, "physfire_train.csv")
        self.master_df = pd.read_csv(csv_path)
        self.samples = self._build_indices()

    def _build_indices(self):
        indices = []
        train_df = self.master_df[self.master_df['region'].isin(self.regions)]
        for _, row in train_df.iterrows():
            ir_rel = row['ir_video_path'].lstrip('./')
            mask_rel = row['mask_video_path'].lstrip('./')
            env_rel = row['env_info_path'].lstrip('./')
            indices.append({
                'ir_path': os.path.join(self.root_dir, ir_rel),
                'mask_path': os.path.join(self.root_dir, mask_rel),
                'env_path': os.path.join(self.root_dir, env_rel)
            })
        print(f"Loaded {len(indices)} aligned samples for {self.regions}")
        return indices

    def _apply_augmentation(self, past_ir, future_ir, future_mask):
        """Consistent geometric transforms across all temporal sequences."""
        # Horizontal Flip
        if random.random() > 0.5:
            past_ir = TF.hflip(past_ir)
            future_ir = TF.hflip(future_ir)
            future_mask = TF.hflip(future_mask)

        # Vertical Flip
        if random.random() > 0.5:
            past_ir = TF.vflip(past_ir)
            future_ir = TF.vflip(future_ir)
            future_mask = TF.vflip(future_mask)

        # Random Rotation (90, 180, 270)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            past_ir = TF.rotate(past_ir, angle)
            future_ir = TF.rotate(future_ir, angle)
            future_mask = TF.rotate(future_mask, angle)

        return past_ir, future_ir, future_mask

    def _read_csv_robust(self, path):
        if not os.path.exists(path):
            file_name = os.path.basename(path)
            for r in ["Region A", "Region B", "Region C", "Region D", "Region E"]:
                alt_path = os.path.join(self.root_dir, r, 'Environmental Info', file_name)
                if os.path.exists(alt_path):
                    path = alt_path
                    break

        for enc in ['utf-8', 'utf-16', 'latin-1', 'cp1252']:
            try:
                return pd.read_csv(path, encoding=enc)
            except:
                continue
        return pd.read_csv(path, encoding='utf-8', encoding_errors='replace')

    def _load_video_frames(self, path, num_frames):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return torch.zeros((num_frames, 1, *self.img_size))

        frames = []
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, self.img_size)
            frames.append(frame)
        cap.release()

        if not frames:
            return torch.zeros((num_frames, 1, *self.img_size))

        while len(frames) < num_frames:
            frames.append(frames[-1])

        out = np.stack(frames, axis=0)[:, np.newaxis, :, :]
        # Normalize to [0, 1]
        return torch.from_numpy(out).float() / 255.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        for attempt in range(min(len(self.samples), 10)):
            current_idx = (idx + attempt) % len(self.samples)
            s = self.samples[current_idx]

            try:
                total_frames = self.frames_past + self.frames_future
                ir_video = self._load_video_frames(s['ir_path'], total_frames)
                mask_video = self._load_video_frames(s['mask_path'], total_frames)

                past_ir = ir_video[:self.frames_past]
                future_ir = ir_video[self.frames_past:]
                future_mask = mask_video[self.frames_past:]

                # Apply Augmentation if enabled
                if self.augment:
                    past_ir, future_ir, future_mask = self._apply_augmentation(
                        past_ir, future_ir, future_mask
                    )

                env_df = self._read_csv_robust(s['env_path'])
                env_df.columns = [c.lower().strip() for c in env_df.columns]

                temp = env_df['temperature'].values[0] if 'temperature' in env_df.columns else \
                       env_df['temp'].values[0] if 'temp' in env_df.columns else 0.0
                hum = env_df['humidity'].values[0] if 'humidity' in env_df.columns else 0.0
                wind = env_df['winddirection'].values[0] if 'winddirection' in env_df.columns else \
                       env_df['wind_dir'].values[0] if 'wind_dir' in env_df.columns else 0.0

                wind_rad = np.deg2rad(float(wind))
                weather = torch.tensor([float(temp), float(hum), np.sin(wind_rad), np.cos(wind_rad)], dtype=torch.float32)

                return past_ir, future_ir, future_mask, weather

            except Exception as e:
                continue 

        return torch.zeros((5, 1, 256, 256)), torch.zeros((5, 1, 256, 256)), \
               torch.zeros((5, 1, 256, 256)), torch.zeros(4)
