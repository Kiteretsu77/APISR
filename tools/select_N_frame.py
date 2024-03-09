import os, sys, shutil

parnet_dir = "/media/hikaridawn/w/AVC_train_all"
save_dir = "/media/hikaridawn/w/AVC_train_select5"
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir)

select_num = 5

gap = 100//select_num

for idx, img_name in enumerate(sorted(os.listdir(parnet_dir))):
    if idx % gap != 0:
        continue

    source_path = os.path.join(parnet_dir, img_name)
    destination_path = os.path.join(save_dir, img_name)
    shutil.copy(source_path, destination_path)