#!/bin/bash
# Set up the path (tmp_usm_dir will be removed; the rest will be kept)
full_patch_source=../APISR_dataset
tmp_usm_dir=GEASR_sharpen
degrade_hr_dataset_name=datasets/train_hr
train_hr_dataset_name=datasets/train_hr_enhanced

# Resize images and prepare usm sharpening in Anime
python scripts/anime_strong_usm.py -i $full_patch_source -o $tmp_usm_dir --outlier_threshold 32

# Crop images to the target HR and degradate_HR dataset
python scripts/crop_images.py -i $output_dir_720p_crop --crop_size 256 -o $degrade_hr_dataset_name
python scripts/crop_usm_only.py -i $tmp_usm_dir -o $train_hr_dataset_name --size 256

# Clean unnecessary file
rm -rf $tmp_usm_dir