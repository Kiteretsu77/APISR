#!/bin/bash

input_dir=../Danbooru_select/
# tmp_dir_720p=AVC_720p
output_dir_crop=../datasets_anime/Danbooru_train
tmp_usm_dir=Danbooru_train_usm


# Resize images and prepare usm sharpening in Anime
# python tools/720P_resize.py -i $input_dir  -o $tmp_dir_720p
python tools/4x_crop.py -i $input_dir -o $output_dir_crop
python scripts/anime_strong_usm.py -i $output_dir_crop -o $tmp_usm_dir --outlier_threshold 32

# Crop images to the target HR and degradate_HR dataset
python scripts/crop_images.py -i $output_dir_crop --crop_size 256 -o datasets/train_hr_Danbooru_train_V3
python scripts/crop_usm_only.py -i $tmp_usm_dir -o datasets/train_hr_Danbooru_train_V3_usm --size 256

# Clean unnecessary file
# rm -rf $tmp_dir_720p
rm -rf $tmp_usm_dir

