#!/bin/bash

# Usage: ./script.sh {folder_path}

folder_path="datasets/final/glomerulo/train/Sclerosis"

if [ -d "$folder_path" ]; then
  find "$folder_path" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.gif" -o -iname "*.bmp" \) | head -n 5
else
  echo "The provided folder path is not valid."
fi