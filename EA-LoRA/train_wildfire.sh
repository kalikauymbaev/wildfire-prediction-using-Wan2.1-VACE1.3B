MODEL_PATH='["/workspace/Wan2.1/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors", "/workspace/Wan2.1/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth", "/workspace/Wan2.1/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth"]'

DATASET_ROOT="/workspace/Wan2.1/FireSentry-Benchmark-Dataset"
METADATA_CSV="$DATASET_ROOT/physfire_train.csv"
TOKENIZER_PATH="/workspace/Wan2.1/Wan2.1-VACE-1.3B/google/umt5-xxl"

accelerate launch --mixed_precision="bf16" train.py \
  --model_paths "$MODEL_PATH" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --dataset_base_path "$DATASET_ROOT" \
  --dataset_metadata_path "$METADATA_CSV" \
  --data_file_keys "ir_video_path" \
  --extra_inputs "ir_video_path,prompt_ir,env_info_path" \
  --height 480 \
  --width 720 \
  --num_frames 17 \
  --max_pixels 518400 \
  --trainable_models "dit" \
  --lora_target_modules "blocks" \
  --lora_rank 64 \
  --learning_rate 2e-5 \
  --gradient_accumulation_steps 4 \
  --num_epochs 50 \
  --output_path "./output/wildfire_ea_lora" \
  --task "sft" \
  --use_gradient_checkpointing
