#!/bin/bash

# This is the example to demonstrate how to download all the XL-LoRA synthesised data

REPO="mbasoz/xllora-datasets"
BASE_URL="https://huggingface.co/datasets/$REPO/resolve/main"

for file in \
  xllora-afrikaans.csv \
  xllora-hausa.csv \
  xllora-hindi.csv \
  xllora-indonesian.csv \
  xllora-korean.csv \
  xllora-marathi.csv \
  xllora-telugu.csv
do
  wget -O "$file" "$BASE_URL/$file"
done