#!/bin/bash

# This is the example to demonstrate the training settings of the Hausa Language Adapter LoRA model

# Create the HF_TOKEN environment variable using the token you get from your Hugging Face account
# Ensure HF_TOKEN is set in the environment
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Please run: export HF_TOKEN=hf_xxx"
  exit 1
fi


output_model_path=../models/lora-gemma327b-la-hausa # output path to save the lora model
base_model_name=google/gemma-3-27b-it
# dataset_lan_other: additional training dataset language if the training data size is less than the size set. It uses the CohereLabs/aya_collection_language_split dataset

python3 -u ../src/lora_training.py \
  -o ${output_model_path} \
  --dataset_name CohereLabs/aya_dataset --dataset_lang hau \
  --dataset_lan_other hausa \
  --language Hausa --target_modules "all-linear" \
  --prepare_model_for_kbit_training --rank 8 \
  --model_name ${base_model_name} \
  --per_device_train_batch_size 16 \
  --train_dataset_size 10000