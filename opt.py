# -*- coding: utf-8 -*-
import os


opt = {}
##################################################### Frequently Changed Setting ###########################################################
opt['description'] = "4x_GRL_paper"        # Description to add to the log  

opt['architecture'] = "GRL"                # "ESRNET" || "ESRGAN" || "GRL" || "GRLGAN" (GRL only support 4x) || "DAT" || "DATGAN"


# Essential Setting
opt['scale'] = 4                                                    # In default, this is 4x
opt["full_patch_source"] = "../datasets_anime/APISR_dataset"        # The HR image without cropping 
opt["degrade_hr_dataset_path"] = "datasets/train_hr"                # The cropped GT images
opt["train_hr_dataset_path"] = "datasets/train_hr_enhanced"         # The cropped Pseudo-GT path (after hand-drawn line enhancement)
############################################################################################################################################

# GPU setting
opt['CUDA_VISIBLE_DEVICES'] = '0'               # '0' / '1' based on different GPU you have or you can use CUDA_VISIBLE_DEVICES=1 for linux
os.environ['CUDA_VISIBLE_DEVICES'] = opt['CUDA_VISIBLE_DEVICES']  


##################################################### Setting for General Training ##########################################################

# Dataset Setting
opt["lr_dataset_path"] = "datasets/train_lr"    # Where you temporally store the LR synthetic images
opt['hr_size'] = 256


# Loss function
opt['pixel_loss'] = "L1"    # Usually it is "L1" 


# Adam optimizer setting
opt["adam_beta1"] = 0.9
opt["adam_beta2"] = 0.99
opt['decay_gamma'] = 0.5    # Decay the learning rate per decay_iteration


# Miscellaneous Setting
opt['degradate_generation_freq'] = 1     # How frequent we degradate HR to LR (1: means Real-Time Degrade) [No need to change this]
opt['train_dataloader_workers'] = 5      # Number of workers for DataLoader
opt['checkpoints_freq'] = 50             # frequency to store checkpoints in the folder (unit: epoch)

#############################################################################################################################################




###################################################### Model Specific Setting ###############################################################

# Add setting for different architecture (Please go through the model architecture you want!)
if opt['architecture'] == "GRL":             # L1 loss training version
    # Setting for GRL Training 
    opt['model_size'] = "tiny2"              # "tiny2" in default
    
    opt['train_iterations'] = 300000         # Training Iterations
    opt['train_batch_size'] = 32             # 4x: 32 (256x256); 2x:  4?  
    
    # Learning Rate
    opt["start_learning_rate"] = 0.0002      # Training Epoch, use the as Real-ESRGAN: 0.0001 - 0.0002 is ok, based on your need
    opt['decay_iteration'] = 100000          # Decay iteration  
    opt['double_milestones'] = []            # Iteration based time you double your learning rate (Just ignore this one)

elif opt['architecture'] == "GRLGAN":        # L1 + Preceptual + Discriminator Loss version
    # Setting for GRL Training
    opt['model_size'] = "tiny2"              # "small" || "tiny" || "tiny2"  (Use tiny2 by default, No need to change)

    # Setting for GRL-GAN Traning
    opt['train_iterations'] = 300000         # Training Iterations
    opt['train_batch_size'] = 32             # 4x: 32 batch size (for 256x256); 2x: 4        
    
    # Learning Rate
    opt["start_learning_rate"] = 0.0001      # Training Epoch, use the as Real-ESRGAN: 0.0001 - 0.0002 is ok, based on your need
    opt['decay_iteration'] = 100000          # Fixed decay gap
    opt['double_milestones'] = []            # Just put this empty
    
    # Perceptual loss
    opt["danbooru_perceptual_loss_weight"] = 0.5        # ResNet50 Danbooru Perceptual loss weight scale
    opt["vgg_perceptual_loss_weight"] = 0.5             # VGG PhotoRealistic Perceptual loss weight scale
    opt['train_perceptual_vgg_type'] = 'vgg19'          # VGG16/19 (Just use 19 by default)
    opt['train_perceptual_layer_weights'] = {'conv1_2': 0.1, 'conv2_2': 0.1, 'conv3_4': 1, 'conv4_4': 1, 'conv5_4': 1}      # Middle-Layer weight for VGG
    opt['Danbooru_layer_weights'] = {"0": 0.1, "4_2_conv3": 20, "5_3_conv3": 25, "6_5_conv3": 1, "7_2_conv3": 1}            # Middle-Layer weight for ResNet
    
    # GAN loss
    opt["discriminator_type"] = "PatchDiscriminator"        # "PatchDiscriminator" || "UNetDiscriminator" 
    opt["gan_loss_weight"] = 0.2                            


elif opt['architecture'] == "ESRNET":
    
    # Setting for ESRNET Training 
    opt['ESR_blocks_num'] = 6                # How many RRDB blocks you need
    opt['train_iterations'] = 500000         # Training Iterations (500K for large resolution large dataset overlap training)
    opt['train_batch_size'] = 32             # 

    # Learning Rate
    opt["start_learning_rate"] = 0.0002      # Training Epoch, use the as Real-ESRGAN: 0.0001 - 0.0002 is ok, based on your need
    opt['decay_iteration'] = 100000          # Decay iteration  
    opt['double_milestones'] = []            # Iteration based time you double your learning rate

elif opt['architecture'] == "ESRGAN":
    
    # Setting for ESRGAN Training 
    opt['ESR_blocks_num'] = 6                # How many RRDB blocks you need
    opt['train_iterations'] = 200000         # Training Iterations
    opt['train_batch_size'] = 32             #      
    
    # Learning Rate
    opt["start_learning_rate"] = 0.0001     # Training Epoch, use the as Real-ESRGAN: 0.0001 - 0.0002 is ok, based on your need
    opt['decay_iteration'] = 100000         # Fixed decay gap
    opt['double_milestones'] = []           # Just put this empty
    
    # Perceptual loss
    opt["danbooru_perceptual_loss_weight"] = 0.5        # ResNet50 Danbooru Perceptual loss weight scale
    opt["vgg_perceptual_loss_weight"] = 0.5             # VGG PhotoRealistic Perceptual loss weight scale
    opt['train_perceptual_vgg_type'] = 'vgg19'          # VGG16/19 (Just use 19 by default)
    opt['train_perceptual_layer_weights'] = {'conv1_2': 0.1, 'conv2_2': 0.1, 'conv3_4': 1, 'conv4_4': 1, 'conv5_4': 1}      # Middle-Layer weight for VGG
    opt['Danbooru_layer_weights'] = {"0": 0.1, "4_2_conv3": 20, "5_3_conv3": 25, "6_5_conv3": 1, "7_2_conv3": 1}            # Middle-Layer weight for ResNet
    
    # GAN loss
    opt["discriminator_type"] = "PatchDiscriminator"        # "PatchDiscriminator" || "UNetDiscriminator" 
    opt["gan_loss_weight"] = 0.2                            # 


elif opt['architecture'] == "DAT":           # L1 loss training version
    
    # Setting for DAT Training 
    opt['model_size'] = "small"              # "light" || "small"
    
    opt['train_iterations'] = 500000         # Training Iterations
    opt['train_batch_size'] = 12             # For 4x, light can have 32 batch size; small can have    batch size
    
    # Learning Rate
    opt["start_learning_rate"] = 0.0001      # Training Epoch, use the as Real-ESRGAN: 0.0001 - 0.0002 is ok, based on your need
    opt['decay_iteration'] = 100000          # Decay iteration  
    opt['double_milestones'] = []            # Iteration based time you double your learning rate (Just ignore this one)

elif opt['architecture'] == "DATGAN":         # L1 + Preceptual + Discriminator Loss version
    
    # Setting for DATGAN Training 
    opt['model_size'] = "small"              # "light" || "small"

    # Setting for DAT-GAN Training
    opt['train_iterations'] = 300000         # Training Iterations
    opt['train_batch_size'] = 12             # 4x: 32 batch size (for 256x256); 2x: 4        
    
    # Learning Rate
    opt["start_learning_rate"] = 0.0001      # Training Epoch, use the as Real-ESRGAN: 0.0001 - 0.0002 is ok, based on your need
    opt['decay_iteration'] = 100000          # Fixed decay gap
    opt['double_milestones'] = []            # Just put this empty
    
    # Perceptual loss
    opt["danbooru_perceptual_loss_weight"] = 0.5        # ResNet50 Danbooru Perceptual loss weight scale
    opt["vgg_perceptual_loss_weight"] = 0.5             # VGG PhotoRealistic Perceptual loss weight scale
    opt['train_perceptual_vgg_type'] = 'vgg19'          # VGG16/19 (Just use 19 by default)
    opt['train_perceptual_layer_weights'] = {'conv1_2': 0.1, 'conv2_2': 0.1, 'conv3_4': 1, 'conv4_4': 1, 'conv5_4': 1}      # Middle-Layer weight for VGG
    opt['Danbooru_layer_weights'] = {"0": 0.1, "4_2_conv3": 20, "5_3_conv3": 25, "6_5_conv3": 1, "7_2_conv3": 1}            # Middle-Layer weight for ResNet
    
    # GAN loss
    opt["discriminator_type"] = "PatchDiscriminator"        # "PatchDiscriminator" || "UNetDiscriminator" 
    opt["gan_loss_weight"] = 0.2                            # 


elif opt['architecture'] == "CUNET":
    # Setting for CUNET Training 
    opt['train_iterations'] = 500000        # Training Iterations (700K for large resolution large dataset overlap training)
    opt['train_batch_size'] = 16          

    opt["start_learning_rate"] = 0.0002     # Training Epoch, use the as Real-ESRGAN: 0.0001 - 0.0002 is ok, based on your need
    opt['decay_iteration'] = 100000         # Decay iteration  
    opt['double_milestones'] = []           # Iteration based time you double your learning rate


elif opt['architecture'] == "CUGAN":
    # Setting for ESRGAN Training 
    opt['ESR_blocks_num'] = 6                # How many RRDB blocks you need
    opt['train_iterations'] = 200000         # Training Iterations
    opt['train_batch_size'] = 16        
    opt["start_learning_rate"] = 0.0001      # Training Epoch, use the as Real-ESRGAN: 0.0001 - 0.0002 is ok, based on your need

    opt["perceptual_loss_weight"] = 1.0
    opt['train_perceptual_vgg_type'] = 'vgg19'
    opt['train_perceptual_layer_weights'] = {'conv1_2': 0.1, 'conv2_2': 0.1, 'conv3_4': 1, 'conv4_4': 1, 'conv5_4': 1}
    opt['Danbooru_layer_weights'] = {"0": 0.1, "4_2_conv3": 20, "5_3_conv3": 25, "6_5_conv3": 1, "7_2_conv3": 1}            # Middle-Layer weight for ResNet
    opt["gan_loss_weight"] = 0.2   # This one is very important, Don't neglect it. Based on the paper, it should be 0.1 scale

    opt['decay_iteration'] = 100000                              # Decay iteration  
    opt['double_milestones'] = []            # Iteration based time you double your learning rate

    
else:
    raise NotImplementedError("Please check you architecture option setting!")
#################################################################################################################################################################




###################################################################### Degradation Setting ######################################################################

# Basic setting for degradation
opt["degradation_batch_size"] = 128                 # Degradation batch size
opt["augment_prob"] = 0.5                           # Probability of augmenting (Flip, Rotate) the HR and LR dataset in dataset loading part                                

  
# Parallel Process
opt['parallel_num'] = 6     # Multi-Processing num; Recommend 6~8 based on your CPU

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
opt['resize_prob'] = [0.2, 0.7, 0.1]            # [up, down, keep] Resize Probability
opt['resize_range'] = [0.1, 1.2]                # Was [0.15, 1.5] in Real-ESRGAN
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
opt['resize_options'] = ['area', 'bilinear', 'bicubic']         # Should be supported by F.interpolate


# First image compression
opt['compression_codec1'] = ["jpeg", "webp", "heif", "avif"]     # Compression codec: heif is the intra frame version of HEVC (H.265) and avif is the intra frame version of AV1
opt['compression_codec_prob1'] = [0.4, 0.6, 0.0, 0.0] 

# Specific Setting
opt["jpeg_quality_range1"] = [20, 95]       # Harder JPEG compression setting
opt["webp_quality_range1"] = [20, 95]
opt["webp_encode_speed1"] = [0, 6]
opt["heif_quality_range1"] = [30, 100]
opt["heif_encode_speed1"] = [0, 6]          # Useless now
opt["avif_quality_range1"] = [30, 100]
opt["avif_encode_speed1"] = [0, 6]          # Useless now


######################################## Setting for Degradation with Intra-Prediction ###############################################################################
opt['compression_codec2'] = ["jpeg", "webp", "avif", "mpeg2", "mpeg4", "h264", "h265"]     # Compression codec: similar to VCISR but more intense degradation settings
opt['compression_codec_prob2'] = [0.06, 0.1, 0.1, 0.12, 0.12, 0.3, 0.2] 

# Image compression setting
opt["jpeg_quality_range2"] = [20, 95]       # Harder JPEG compression setting

opt["webp_quality_range2"] = [20, 95]
opt["webp_encode_speed2"] = [0, 6]

opt["avif_quality_range2"] = [20, 95]
opt["avif_encode_speed2"] = [0, 6]          # Useless now

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

####################################################################################################################################################################

