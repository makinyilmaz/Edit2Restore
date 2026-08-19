#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1

python train_lora_flux_kontext_multiple.py \
  --pretrained_model_name_or_path "black-forest-labs/FLUX.1-Kontext-dev" \
  --degradation_folders "train_data32/denoising" \
                        "train_data32/deraining" \
                        "train_data32/dehazing" \
                        "train_data32/decompression" \
                        "train_data32/lowlight" \
  --degradation_prompts "Remove noise and grain, restore a clean and sharp image" \
                        "Remove rain streaks and restore the clean image" \
                        "Remove haze and fog, restore clear visibility and natural colors" \
                        "Remove compression artifacts and restore the clean image" \
                        "Brighten this photo, recover shadow details, reduce noise, keep colors natural" \
  --resolution 1024 \
  --train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 20 \
  --learning_rate 1e-4 \
  --rank 64 \
  --gradient_checkpointing \
  --mixed_precision bf16 \
  --allow_tf32 \
  --output_dir "trained_loras/fluxkontextdev/all-in-one/sample32_rank64_textencoder" \
  --train_text_encoder \
  --text_encoder_lr 5e-6 \
  --repeats 4 \
  --seed 42
