Capstone Project Task: Wildfire Expansion Prediction using WorldModel

Applied Task: Wildfire Expansion Prediction using World-Foundational model Wan2.1- VACE1.3B

Applied model: Wan2.1-VACE1.3B

Dataset: FireSentry Dataset

Dataset implementation: We have created a .csv file where we have 5 columns(region, ir_video_path, mask_video_path, env_info_path, prompt_it, prompt_mask). This file is created for training the model and read file from here and seek it in the dataset folder

Applied Methodologies: We did experiments with 2 methodologies to train the model which are: 
1) Environmental-Aware LoRA(EA-LoRA);
2) Wan2.1-VACE1.3B + LoRA + weather encoder + future IR forecasting + mask prediction

Applied System Requirements: We have used remote GPU NVIDIA A100 with 40/80 GB VRAM, 350 GB memory on vast.ai website

Used Framework: DiffSynth-Studio for LoRA technique

Logic for this task: Out of the 5 available geographical regions, 4 were used for active training (Regions A, B, C, and E), while the 5th region (Region D) was strictly reserved for "Hold-out" testing

This repository contains main updates, generated videos, images, modified or created files to implement this task. Not all source code was uploaded due to the limit of github repository
