#!/bin/bash

# This is the example to demonstrate the sentence embedding training for Hausa language

# Create the HF_TOKEN environment variable using the token you get from your Hugging Face account
# Ensure HF_TOKEN is set in the environment
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Please run: export HF_TOKEN=hf_xxx"
  exit 1
fi

model_output_dir=../models/sentence-embedding-xlmr-xllora-hausa # output dir path to save the model
training_data_file=../data/xllora-synthesised-gemma327b-hausa.csv # The script expects sent0 (anchor sentence), sent1 (positive sentence) and hard_neg (hard negative sentence) columns in the training dataset
model_name=FacebookAI/xlm-roberta-base

python3 ../src/simcse_train.py \
  --model_name_or_path ${model_name} \
  --output_dir ${model_output_dir} \
  --train_file ${training_data_file} \
  --pooler_type avg_first_last \
  --do_mlm False \
  --save_steps 125 \
  --save_strategy steps \
  --save_total_limit 1 \
  --greater_is_better True \
  --num_train_epochs 3 \
  --eval_strategy steps \
  --eval_language_type hau \
  --eval_type str_other \
  --split_type dev \
  --per_device_train_batch_size 512 \
  --learning_rate 5e-5 \
  --max_seq_length 32 \
  --load_best_model_at_end False \
  --eval_steps 125 \
  --temp 0.05 \
  --report_to none \
  --metric_for_best_model eval_spearman_score \
  --overwrite_output_dir True \
  --do_train True \
  --fp16 True

