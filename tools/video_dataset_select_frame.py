'''
    The purpose of this file is to select first, second, and the last frame from the video datasets.
'''

import os, sys, shutil, cv2

dirs = [
    # "../datasets/VideoLQ",
    # "../datasets/REDS_blur_MPEG",
    "../datasets_real/AVC-RealLQ",
]
store_dirs = [
    # "../datasets/VideoLQ_select",
    # "../datasets/REDS_blur_MPEG_select",
    "AVC",
]
crop_large_img = True   # If the image is larger than 720p, we will first crop them
assert(len(dirs) == len(store_dirs))



# Iterate each dataset
for idx, parent_dir in enumerate(dirs):
    print("This dir is ", parent_dir)

    # Make new dir 
    store_dir = store_dirs[idx]
    if os.path.exists(store_dir):
        shutil.rmtree(store_dir)
    os.makedirs(store_dir)

    # Iterate to Sub Folder sequence
    for sub_folder in sorted(os.listdir(parent_dir)):
        folder_dir = os.path.join(parent_dir, sub_folder)

        # Find all image paths
        image_paths = []
        for img_name in sorted(os.listdir(folder_dir)):
            if img_name.split('.')[-1] in ['jpg', 'png']:
                # Sometimes the folder may contain unneeded info, we don't consider them
                image_paths.append(img_name)
        image_paths = sorted(image_paths)

        # Find three frames (First, Middle, Last)
        first, middle, last = image_paths[0], image_paths[len(image_paths)//2], image_paths[-1]
        print("First, Middle, Last image name is ", first, middle, last)
        
        # Save the three images
        for img_name in [first, middle, last]:
            input_name = os.path.join(folder_dir, img_name)
            
            img = cv2.imread(input_name)
            h, w, _ = img.shape
            if crop_large_img and h*w > 720*1080:
                # This means that this image is too big we need to crop them
                print("We will use cropping for images that is too large")
                crop1 = img[:,:w//2,:]
                crop2 = img[:,w//2:,:]

                store_name1 = os.path.join(store_dir, sub_folder + "_crop1_"+ img_name)
                store_name2 = os.path.join(store_dir, sub_folder + "_crop2_"+ img_name)

                cv2.imwrite(store_name1, crop1)
                cv2.imwrite(store_name2, crop2)
            else:
                store_name = os.path.join(store_dir, sub_folder + "_" + img_name)
                shutil.copy(input_name, store_name)
