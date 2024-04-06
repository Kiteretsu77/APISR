#!/bin/bash

# Set up the PATH TO THE SOURCE
input_source=../datasets_anime/APISR_dataset/

# The following three paths will be used widely in opt.py setting!!!
uncropped_hr=datasets/APISR_720p_4xcrop
degrade_hr_dataset_path=datasets/train_hr
train_hr_dataset_path=datasets/train_hr_enhanced

# tmp path (No need to change, we will remove them at the end of process)
tmp_dir_720p=APISR_720p_tmp
tmp_enhanced_dir=APISR_sharpen_tmp


# Resize images and prepare usm sharpening in Anime
python tools/720P_resize.py -i $input_source  -o $tmp_dir_720p
python tools/4x_crop.py -i $tmp_dir_720p -o $uncropped_hr
python scripts/anime_strong_usm.py -i $uncropped_hr -o $tmp_enhanced_dir --outlier_threshold 32 --num_workers 6


# Crop images to the target HR and degradate_HR dataset
python scripts/crop_images.py -i $uncropped_hr --crop_size 256 -o $degrade_hr_dataset_path
python scripts/crop_images.py -i $tmp_enhanced_dir --crop_size 256 -o $train_hr_dataset_path


# Clean unnecessary file
rm -rf $tmp_dir_720p
rm -rf $tmp_enhanced_dir