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

Installation:
1) git clone https://github.com/Wan-Video/Wan2.1.git
cd Wan2.1
2) git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
3) python -m venv wanvenv
source wanvenv/bin/activate
4) First, install core build tools and the Numpy version you need
pip install ninja packaging setuptools wheel "numpy>=1.23.5,<2"
Install the main research stack
pip install "torch>=2.4.0" "torchvision>=0.19.0" "di`users>=0.31.0" \
"transformers>=4.49.0" "tokenizers>=0.20.3" "accelerate>=1.1.1" \
"gradio>=5.0.0" "opencv-python>=4.9.0.80" \
tqdm imageio easydict ftfy dashscope imageio-`mpeg
# Set paths for your specific CUDA 13.0 environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$PATH
# Use 12 parallel processes for a fast, stable build
export MAX_JOBS=12
pip install flash-attn --no-build-isolation
5) hf download Wan-AI/Wan2.1-VACE-1.3B --local-dir ./Wan2.1-VACE-1.3B
Example of generation video on Wan2.1-VACE1.3B:
python generate.py --task vace-1.3B --size 832*480 --ckpt_dir ./Wan2.1-VACE-1.3B --
src_ref_images examples/girl.png,examples/snake.png --prompt "在一个欢乐而充满节日
气氛的场景中，穿着鲜艳红色春服的小
孩正与她的可爱卡通蛇嬉戏。她的春服上绣着金色吉祥图案，散发着喜庆的气息，脸上
洋溢着灿烂的笑容。蛇身呈现出亮眼的绿色，形状圆润，宽大的眼睛让它显得既友善又幽
默。小女孩欢快地用手轻轻抚摸着蛇的头部，共同享受着这温馨的时刻。周围五彩斑斓的
灯笼和彩带装饰着环境
阳光透过洒在她们身上，营造出一个充满友爱与幸福的新年氛围。
"

This repository contains main updates, generated videos, images, modified or created files to implement this task. Not all source code was uploaded due to the limit of github repository
