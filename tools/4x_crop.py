import os, sys, cv2, shutil, argparse



if __name__ == "__main__":

    # Parse variables available
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type = str)
    parser.add_argument('-o', '--store_dir', type = str)
    args  = parser.parse_args()

    input_dir = args.input_dir
    store_dir = args.store_dir


    print("We are cropping the image for 4x scale such that it is suitable for video compression")

    # Check file
    if os.path.exists(store_dir):
        shutil.rmtree(store_dir)
    os.makedirs(store_dir)

    # Process
    for file_name in sorted(os.listdir(input_dir)):
        need_reszie = False
        source_path = os.path.join(input_dir, file_name)
        destination_path = os.path.join(store_dir, file_name)

        img = cv2.imread(source_path)
        h, w, c = img.shape

        if h % 8 != 0:
            print("We need vertical resize")
            need_reszie = True
            img = img[:8 * (h // 8), :, :]
        
        if w % 8 != 0:
            print("We need horizontal resize")
            need_reszie = True
            img = img[:, :8 * (w // 8), :]
        

        if need_reszie:
            cv2.imwrite(destination_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            shutil.copy(source_path, destination_path)