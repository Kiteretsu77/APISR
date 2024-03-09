'''
    This file is to help us make figure of CRF vs Preset in Video Compression
'''
import os, sys, shutil

def compress(input_folder, codec, crf, preset):
    video_store_name = "compressed.mp4"
    store_dir = input_folder + "_crf" + str(crf) + "_" + preset
    
    if os.path.exists(video_store_name):
        os.remove(video_store_name)
    if os.path.exists(store_dir):
        shutil.rmtree(store_dir)
    os.makedirs(store_dir)


    # Encode 
    os.system("ffmpeg -r 30 -f image2 -i " + input_folder + "/%d.png -vcodec " + codec + " -crf " + str(crf) + " -preset " + preset + " -pix_fmt yuv420p " + video_store_name)

    # Split to frames
    os.system("ffmpeg -i " + video_store_name + " " + store_dir + "/test_%06d.png")



if __name__ == "__main__":
    input_folders = ["ReadySetGo", "Jockey"]
    codec = "libx264"
    crf_ranges = [25 + 5*i for i in range(6)]
    preset_ranges = ["ultrafast", "veryfast", "fast", "medium", "slow", "veryslow", "placebo"]


    for input_folder in input_folders:
        for crf in crf_ranges:
            for preset in preset_ranges:
                print("We are handling {} with crf {} with preset {}".format(input_folder, crf, preset))
                compress(input_folder, codec, crf, preset)

        