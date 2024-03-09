import os, shutil

dir = "../datasets/VideoLQ"
store_dir = "../datasets/VideoLQ_select_one"
if os.path.exists(store_dir):
    shutil.rmtree(store_dir)
os.makedirs(store_dir)


search_idx = 0
for sub_folder_name in sorted(os.listdir(dir)):
    sub_folder_dir = os.path.join(dir, sub_folder_name)
    for idx, img_name in enumerate(sorted(os.listdir(sub_folder_dir))):
        if idx != search_idx:
            continue
        img_path = os.path.join(sub_folder_dir, img_name)
        target_path = os.path.join(store_dir, img_name)

        shutil.copy(img_path, target_path)

    search_idx += 1