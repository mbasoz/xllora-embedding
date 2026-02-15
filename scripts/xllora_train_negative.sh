#!/bin/bash

# This is the example to demonstrate the training settings of the XL-LoRA hard negative examples model

# Create the HF_TOKEN environment variable using the token you get from your Hugging Face account
# Ensure HF_TOKEN is set in the environment
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Please run: export HF_TOKEN=hf_xxx"
  exit 1
fi

output_model_path=../models/lora-gemma327b-xllora-neg # output path to save the lora model
local_train_dataset_path=../data/mixed_parallel_xnli_14l_opusmt_10k_fin_neg.csv # The lora_training.py script expects sent0 (anchor sentence), hard_neg (negative sentence) columns in the dataset
base_model_name=google/gemma-3-27b-it

python3 -u ../src/lora_training.py \
  -o ${output_model_path} \
  --local_dataset --train_dataset_path ${local_train_dataset_path} \
  --language English --target_modules "all-linear" \
  --prepare_model_for_kbit_training --rank 8 \
  --model_name ${base_model_name} \
  --per_device_train_batch_size 16 \
  --completion_only_loss \
  --train_dataset_size 10000 \
  --completion_dataset