#!/bin/bash

python train_lora_flux_kontext_multiple.py \
  --pretrained_model_name_or_path=black-forest-labs/FLUX.1-Kontext-dev \
  --degradation_folders "train_data128/denoising" \
                        "train_data128/deraining" \
                        "train_data128/dehazing" \
  --degradation_prompts "remove the noise from the image" \
                        "remove the rain from the image" \
                        "remove the haze from the image" \
  --resolution 1024 \
  --train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 20 \
  --learning_rate 1e-4 \
  --rank 64 \
  --gradient_checkpointing \
  --mixed_precision bf16 \
  --allow_tf32 \
  --output_dir "lora_model_textenc128" \
  --train_text_encoder \
  --text_encoder_lr 5e-6 \
  --repeats 1 \
  --seed 42
