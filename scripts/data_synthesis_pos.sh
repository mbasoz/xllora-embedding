#!/bin/bash

# This is the example to demonstrate the data synthesis settings for the positive examples

# Create the HF_TOKEN environment variable using the token you get from your Hugging Face account
# Ensure HF_TOKEN is set in the environment
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Please run: export HF_TOKEN=hf_xxx"
  exit 1
fi

output_data_path=../data/xllora-syn-hausa-pos.csv # output csv path to save the data
input_dataset_path=../data/sent0_hausa.csv # The script expects sent0 (anchor sentence) column in the input dataset
base_model_name=google/gemma-3-27b-it
lora_model=../models/lora-gemma327b-xllora-pos
# sample_size: number of examples to synthesise. This number is limited to the input_dataset_path example size

python3 -u ../src/generate_answers_mgpu_orch.py -o ${output_data_path} \
  --model_name ${base_model_name} \
  --peft_model_name ${lora_model} \
  --input_dataset_path ${input_dataset_path} \
  --inference_batch_size 128 \
  --max_seq_length 256 \
  --sample_size 1000 \
  --language English \
  --number_of_gpus_dedicated 1 \
  --positive_prompts # this boolean value sets the generation prompt for positive examples. If this value is not set, the default mode is the negative example synthesis
