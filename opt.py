# -*- coding: utf-8 -*-
import os


opt = {}
##################################################### Global Setting ###########################################################
opt['description'] = "4x_GRL_paper"             # Description to add to the log  

opt['architecture'] = "GRL"                     # "GRL" || "GRLGAN"  

################################################################################################################################

# GPU setting
opt['CUDA_VISIBLE_DEVICES'] = '0'           #   '0/1'
os.environ['CUDA_VISIBLE_DEVICES'] = opt['CUDA_VISIBLE_DEVICES']  


##################################################### Setting for General Training #############################################
# Essential setting
opt['scale'] = 4                         # In default, this is 4x
opt['degradate_generation_freq'] = 1     # How frequent we degradate HR to LR (1: means Real-Time Degrade) [No need to change this]
opt['train_dataloader_workers'] = 5      # Number of workers for DataLoader
opt['checkpoints_freq'] = 50             # frequency to store checkpoints in the folder (unit: epoch)

# Dataset Path
opt["full_patch_source"] = "../datasets_anime/APISR_dataset"        # The true input, we will use path to do generate lr in every epoch. Please read the README of the training 
opt["degrade_hr_dataset_name"] = "datasets/train_hr"                # The GT path
opt["train_hr_dataset_name"] = "datasets/train_hr_enhanced"         # The Pseudo-GT path (after hand-drawn line enhancement)
opt["lr_dataset_name"] = "datasets/train_lr"                        # Where you temporally store the LR generated result
opt['hr_size'] = 256


# Loss function
opt['pixel_loss'] = "L1"                                # Usually it is "L1" 


# Adam optimizer setting
opt["adam_beta1"] = 0.9
opt["adam_beta2"] = 0.99
opt['decay_gamma'] = 0.5                                # Decay the learning rate per decay_iteration

#################################################################################################################################


if opt['architecture'] == "GRL":        # L1 loss training version
    # Setting for GRL Training 
    opt['model_size'] = "tiny2"               # "small" || "tiny" || "tiny2"

    opt['train_iterations'] = 300000         # Training Iterations
    opt['train_batch_size'] = 32             # 4x: 32 (256x256); 2x:  4?  
    opt['use_pretrained'] = False            # If we want to use pretrained weight
    opt["start_learning_rate"] = 0.0002      # Training Epoch, use the as Real-ESRGAN: 0.0001 - 0.0002 is ok, based on your need

    opt['decay_iteration'] = 100000          # Decay iteration  
    opt['double_milestones'] = []            # Iteration based time you double your learning rate (Just ignore this one)


elif opt['architecture'] == "GRLGAN":   # L1 + Preceptual + Discriminator Loss version
    # Setting for GRL Training
    opt['model_size'] = "tiny2"               # "small" || "tiny" || "tiny2"  (Use tiny2 by default, No need to change)

    # Setting for GRL-GAN Traning
    opt['train_iterations'] = 300000         # Training Iterations
    opt['train_batch_size'] = 32             # 4x: 32 batch size (for 256x256); 2x: 4        
    opt['use_pretrained'] = True             # If we want to use pretrained weight (name: grlgan_pretrained.pth)     
    opt["start_learning_rate"] = 0.0001      # Training Epoch, use the as Real-ESRGAN: 0.0001 - 0.0002 is ok, based on your need

    # Perceptual loss
    opt["danbooru_perceptual_loss_weight"] = 0.5        # ResNet50 Danbooru Perceptual loss weight scale
    opt["vgg_perceptual_loss_weight"] = 0.5             # VGG PhotoRealistic Perceptual loss weight scale
    opt['train_perceptual_vgg_type'] = 'vgg19'          # VGG16/19 (Just use 19 by default)
    opt['train_perceptual_layer_weights'] = {'conv1_2': 0.1, 'conv2_2': 0.1, 'conv3_4': 1, 'conv4_4': 1, 'conv5_4': 1}      # Middle-Layer weight for VGG
    opt['Danbooru_layer_weights'] = {"0": 0.1, "4_2_conv3": 20, "5_3_conv3": 25, "6_5_conv3": 1, "7_2_conv3": 1}            # Middle-Layer weight for ResNet
    
    # GAN loss
    opt["discriminator_type"] = "PatchDiscriminator"        # "PatchDiscriminator" || "UNetDiscriminator" 
    opt["gan_loss_weight"] = 0.2                           # 

    opt['decay_iteration'] = 100000       # Fixed decay gap
    opt['double_milestones'] = []         # Just put this empty

else:
    raise NotImplementedError("Please check you architecture option setting!")


# Basic setting for degradation
opt["degradation_batch_size"] = 128                 # Degradation batch size
opt["augment_prob"] = 0.5                           # Probability of augmenting (Flip, Rotate) the HR and LR dataset in dataset loading part                                


if opt['architecture'] in ["ESRNET", "ESRGAN", "GRL", "GRLGAN", "CUNET", "CUGAN"]:        # 这里包含mixed BSR（所以mixed的时候要用ESRNET才不会出bug）
    # Parallel Process
    opt['parallel_num'] = 6  # Multi-Processing num; Recommend 6

    # Blur kernel1
    opt['kernel_range'] = [3, 11]      
    opt['kernel_list'] = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
    opt['kernel_prob'] = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]  
    opt['sinc_prob'] = 0.1             
    opt['blur_sigma'] = [0.2, 3]      
    opt['betag_range'] = [0.5, 4]       
    opt['betap_range'] = [1, 2]      

    # Blur kernel2 
    opt['kernel_list2'] = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
    opt['kernel_prob2'] = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]   
    opt['sinc_prob2'] = 0.1            
    opt['blur_sigma2'] = [0.2, 1.5]    
    opt['betag_range2'] = [0.5, 4]      
    opt['betap_range2'] = [1, 2]        

    # The first degradation process
    opt['resize_prob'] = [0.2, 0.7, 0.1]       
    opt['resize_range'] = [0.1, 1.2]               # Was [0.15, 1.5] in Real-ESRGAN
    opt['gaussian_noise_prob'] = 0.5            
    opt['noise_range'] =  [1, 30]               
    opt['poisson_scale_range'] = [0.05, 3]    
    opt['gray_noise_prob'] =  0.4               
    opt['jpeg_range'] = [30, 95]              

    # The second degradation process
    opt['second_blur_prob'] =  0.8             
    opt['resize_prob2'] = [0.2, 0.7, 0.1]           # [up, down, keep] Resize Probability
    opt['resize_range2'] = [0.15, 1.2]               
    opt['gaussian_noise_prob2'] = 0.5          
    opt['noise_range2'] = [1, 25]               
    opt['poisson_scale_range2'] = [0.05, 2.5]    
    opt['gray_noise_prob2'] = 0.4           
    
    # Other common settings
    opt['resize_options'] = ['area', 'bilinear', 'bicubic']     # Should be supported by F.interpolate


    # First image compression
    opt['compression_codec1'] = ["jpeg", "webp", "heif", "avif"]     # Compression codec: heif is the intra frame version of HEVC (H.265) and avif is the intra frame version of AV1
    opt['compression_codec_prob1'] = [0.4, 0.6, 0.0, 0.0] 

    # Specific Setting
    opt["jpeg_quality_range1"] = [20, 95]
    opt["webp_quality_range1"] = [20, 95]
    opt["webp_encode_speed1"] = [0, 6]
    opt["heif_quality_range1"] = [30, 100]
    opt["heif_encode_speed1"] = [0, 6]       # Useless now
    opt["avif_quality_range1"] = [30, 100]
    opt["avif_encode_speed1"] = [0, 6]       # Useless now

    
    ######################################## Setting for Degradation with Intra-Prediction ########################################
    opt['compression_codec2'] = ["jpeg", "webp", "avif", "mpeg2", "mpeg4", "h264", "h265"]     # Compression codec: similar to VCISR but more intense degradation
    opt['compression_codec_prob2'] = [0.06, 0.1, 0.1, 0.12, 0.12, 0.3, 0.2] 

    # Image compression setting
    opt["jpeg_quality_range2"] = [20, 95]

    opt["webp_quality_range2"] = [20, 95]
    opt["webp_encode_speed2"] = [0, 6]

    opt["avif_quality_range2"] = [20, 95]
    opt["avif_encode_speed2"] = [0, 6]       # Useless now

    # Video compression I-Frame setting
    opt['h264_crf_range2'] = [23, 38]
    opt['h264_preset_mode2'] = ["slow", "medium", "fast", "faster", "superfast"]
    opt['h264_preset_prob2'] = [0.05, 0.35, 0.3, 0.2, 0.1]

    opt['h265_crf_range2'] = [28, 42]
    opt['h265_preset_mode2'] = ["slow", "medium", "fast", "faster", "superfast"]
    opt['h265_preset_prob2'] = [0.05, 0.35, 0.3, 0.2, 0.1]

    opt['mpeg2_quality2'] = [8, 31]         #  linear scale 2-31 (the lower the higher quality)
    opt['mpeg2_preset_mode2'] = ["slow", "medium", "fast", "faster", "superfast"]
    opt['mpeg2_preset_prob2'] = [0.05, 0.35, 0.3, 0.2, 0.1]

    opt['mpeg4_quality2'] = [8, 31]         #  should be the same as mpeg2_quality2
    opt['mpeg4_preset_mode2'] = ["slow", "medium", "fast", "faster", "superfast"]
    opt['mpeg4_preset_prob2'] = [0.05, 0.35, 0.3, 0.2, 0.1]
    

