import os, cv2, shutil, argparse


if __name__ == "__main__":

    # Parse variables available
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type = str)
    parser.add_argument('-o', '--store_dir', type = str)
    args  = parser.parse_args()

    input_dir = args.input_dir
    store_dir = args.store_dir

    print("We are doing the 720p Resize check!")

    # File Check
    if os.path.exists(store_dir):
        shutil.rmtree(store_dir)
    os.makedirs(store_dir)

    scale = 4
    num = 0
    for file_name in sorted(os.listdir(input_dir)):
        source_path = os.path.join(input_dir, file_name)
        destination_path = os.path.join(store_dir, file_name)
        img = cv2.imread(source_path)
        h,w,c = img.shape

        if h == 720:
            # It is already 720P so we directly move them
            shutil.copy(source_path, destination_path)
            continue
        elif h < 720:
            print("It is weird that there is an image with height less than 720 ", file_name)
            break
        
        # Else, here we need to resize them (All resize to 720P)

        new_w = int(w*(720/h))
        img_bicubic = cv2.resize(img, (new_w, 720), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(store_dir, file_name), img_bicubic, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    print("The total resize num is ", num)