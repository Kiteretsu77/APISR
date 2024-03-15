'''
    This file is the whole dataset curation pipeline to collect the least compressed and the most informative frames from video source.
'''
import os, time, sys
import shutil
import cv2
import torch
import argparse

# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from opt import opt
from dataset_curation_pipeline.IC9600.gene import infer_one_image
from dataset_curation_pipeline.IC9600.ICNet import ICNet


class video_scoring:
    
    def __init__(self, IC9600_pretrained_weight_path) -> None:

        # Init the model
        self.scorer = ICNet()
        self.scorer.load_state_dict(torch.load(IC9600_pretrained_weight_path, map_location=torch.device('cpu')))
        self.scorer.eval().cuda()


    def select_frame(self, skip_num, img_lists, target_frame_num, save_dir, output_name_head, partition_idx):
        ''' Execution of scoring to all I-Frame in img_folder and select target_frame to return back
        Args:
            skip_num (int):         Only 1 in skip_num will be chosen to accelerate.
            img_lists (str):        The image lists of all files we want to process
            target_frame_num (int): The number of frames we need to choose
            save_dir (str):         The path where we save those images
            output_name_head (str): This is the input video name head
            partition_idx (int):    The partition idx
        '''

        stores = []
        for idx, image_path in enumerate(sorted(img_lists)):
            if idx % skip_num != 0:
                # We only process 1 in 3 to accelerate and also prevent minor case of repeated scene.
                continue


            # Evaluate the image complexity score for this image
            score = infer_one_image(self.scorer, image_path)

            if verbose:
                print(image_path, score)
            stores.append((score, image_path))

            if verbose:
                print(image_path, score)
        

        # Find the top most scores' images
        stores.sort(key=lambda x:x[0])
        selected = stores[-target_frame_num:]
        # print(len(stores), len(selected))
        if verbose:
            print("The lowest selected score is ", selected[0])     # This is a kind of info


        # Store the selected images
        for idx, (score, img_path) in enumerate(selected):
            output_name = output_name_head + "_" +str(partition_idx)+ "_" + str(idx) + ".png" 
            output_path = os.path.join(save_dir, output_name)
            shutil.copyfile(img_path, output_path)


    def run(self, skip_num, img_folder, target_frame_num, save_dir, output_name_head, partition_num):
        ''' Execution of scoring to all I-Frame in img_folder and select target_frame to return back
        Args:
            skip_num (int):         Only 1 in skip_num will be chosen to accelerate.
            img_folder (str):       The image folder of all I-Frames we need to process
            target_frame_num (int): The number of frames we need to choose
            save_dir (str):         The path where we save those images
            output_name_head (str): This is the input video name head
            partition_num (int):    The number of partition we want to crop the video to
        '''
        assert(target_frame_num%partition_num == 0)

        img_lists = []
        for img_name in sorted(os.listdir(img_folder)):
            path = os.path.join(img_folder, img_name)
            img_lists.append(path)
        length = len(img_lists)
        unit_length = (length // partition_num)
        target_partition_num = target_frame_num // partition_num

        # Cut the folder to several partition and select those with the highest score
        for idx in range(partition_num):
            select_lists = img_lists[unit_length*idx : unit_length*(idx+1)]
            self.select_frame(skip_num, select_lists, target_partition_num, save_dir, output_name_head, idx)


class frame_collector:
    
    def __init__(self, IC9600_pretrained_weight_path, verbose) -> None:
        
        self.scoring = video_scoring(IC9600_pretrained_weight_path)
        self.verbose = verbose


    def video_split_by_IFrame(self, video_path, tmp_path):
        ''' Split the video to its I-Frames format
        Args:
            video_path (str):       The directory to a single video
            tmp_path (str):         A temporary working places to work and will be delete at the end
        '''

        # Prepare the work folder needed
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)
        os.makedirs(tmp_path)
        

        # Split Video I-frame
        cmd = "ffmpeg -i " + video_path + " -loglevel error -vf select='eq(pict_type\,I)' -vsync 2 -f image2 -q:v 1 " + tmp_path + "/image-%06d.png"  # At most support 100K I-Frames per video

        if self.verbose:
            print(cmd)
        os.system(cmd)
        


    def collect_frames(self, video_folder_dir, save_dir, tmp_path, skip_num, target_frames, partition_num):
        ''' Automatically collect frames from the video dir
        Args:
            video_folder_dir (str):     The directory of all videos input
            save_dir (str):             The directory we will store the selected frames
            tmp_path (str):             A temporary working places to work and will be delete at the end
            skip_num (int):             Only 1 in skip_num will be chosen to accelerate.
            target_frames (list):       [# of frames for video under 30 min, # of frames for video over 30 min] 
            partition_num (int):        The number of partition we want to crop the video to   
        '''

        # Iterate all video under video_folder_dir
        for video_name in sorted(os.listdir(video_folder_dir)):
            # Sanity check for this video file format
            info = video_name.split('.')
            if info[-1] not in ['mp4', 'mkv', '']:
                continue
            output_name_head, extension = info


            # Get info of this video
            video_path = os.path.join(video_folder_dir, video_name)
            duration = get_duration(video_path)     # unit in minutes
            print("We are processing " + video_path + " with duration " + str(duration) + " min")


            # Split the video to I-frame
            self.video_split_by_IFrame(video_path, tmp_path)


            # Score the frames and select those top scored frames we need
            if duration <= 30:
                target_frame_num = target_frames[0]
            else:
                target_frame_num = target_frames[1]
            
            self.scoring.run(skip_num, tmp_path, target_frame_num, save_dir, output_name_head, partition_num)


            # Remove folders if needed


def get_duration(filename):
    video = cv2.VideoCapture(filename)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    seconds = frame_count / fps
    minutes = int(seconds / 60)
    return minutes


if __name__ == "__main__":

    # Fundamental setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_folder_dir', type = str, default = '../anime_videos',                  help = "A folder with video sources")
    parser.add_argument('--IC9600_pretrained_weight_path', type = str, default = "pretrained/ck.pth",   help = "The pretrained IC9600 weight")
    parser.add_argument('--save_dir', type = str, default = 'APISR_dataset',                         help = "The folder to store filtered dataset")
    parser.add_argument('--skip_num', type = int, default = 5,                                          help = "Only 1 in skip_num will be chosen in sequential I-frames to accelerate.")
    parser.add_argument('--target_frames', type = list, default = [16, 24],                             help = "[# of frames for video under 30 min, # of frames for video over 30 min]")
    parser.add_argument('--partition_num', type = int, default = 8,                                     help = "The number of partition we want to crop the video to, to increase diversity of sampling")
    parser.add_argument('--verbose', type = bool, default = True,                                       help = "Whether we print log message")
    args  = parser.parse_args()


    # Transform to variable
    video_folder_dir = args.video_folder_dir
    IC9600_pretrained_weight_path = args.IC9600_pretrained_weight_path
    save_dir = args.save_dir
    skip_num = args.skip_num
    target_frames = args.target_frames  # [# of frames for video under 30 min, # of frames for video over 30 min]    
    partition_num = args.partition_num
    verbose = args.verbose


    # Secondary setting
    tmp_path = "tmp_dataset"


    # Prepare
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)


    # Process
    start = time.time()

    obj = frame_collector(IC9600_pretrained_weight_path, verbose)
    obj.collect_frames(video_folder_dir, save_dir, tmp_path, skip_num, target_frames, partition_num)

    total_time = (time.time() - start)//60
    print("Total time spent is {} min".format(total_time))

    shutil.rmtree(tmp_path)