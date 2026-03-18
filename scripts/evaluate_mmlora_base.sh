#!/bin/bash

# This is the example to demonstrate the evaluation of the mmbert pretrained model

# Create the HF_TOKEN environment variable using the token you get from your HuggingFace account
# Ensure HF_TOKEN is set in the environment
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Please run: export HF_TOKEN=hf_xxx"
  exit 1
fi


output_dir_path=../data/mteb_results/
model_name=jhu-clsp/mmBERT-base
tokenizer_name=jhu-clsp/mmBERT-base # Tokenizer should match with the model's original tokenizer
task_name=SemRel24STS # Other alternatives are STS (Korean), MIRACLRetrievalHardNegatives, IndicQARetrieval, BelebeleRetrieval

python3 ../src/evaluation_mteb.py \
  --model_name ${model_name} \
  --task_name ${model_name} \
  --language hau \
  --tokenizer_name ${model_name} \
  --pooler_type avg_first_last \
  --output_dir ${model_name}