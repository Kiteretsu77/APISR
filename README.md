<p align="center">
    <img src="__assets__/logo.png" height="100">
</p>

## APISR: Anime Production Inspired Real-World Anime Super-Resolution (CVPR 2024)
APISR aims at restoring and enhancing low-quality low-resolution anime images and video sources with various degradations from real-world scenarios. 
 
[![Arxiv](https://img.shields.io/badge/Arxiv-<COLOR>.svg)](https://arxiv.org/abs/2403.01598)<br>

👀 [**Visualization**](#Visualization)  **|** 🔥 [Update](#Update) **|** 🔧 [Installation](#installation) **|** 🏰 [**Model Zoo**](docs/model_zoo.md) **|** ⚡ [Inference](#inference) **|** 🧩 [Dataset Curation](#dataset_curation) **|** 💻 [Train](#train)


<p align="center">
    <img src="__assets__/workflow.png" style="border-radius: 15px">
</p>


:star: If you like APISR, please help star this repo. Thanks! :hugs:



<!---------------------------------------- Visualization ---------------------------------------->
## <a name="Visualization"></a> Visualization (Click them for the best view!) 👀

<!-- Kiteret: https://imgsli.com/MjQ1NzE0 -->
<!-- EVA: https://imgsli.com/MjQ1NzIx -->
<!-- Pokemon: https://imgsli.com/MjQ1NzIy -->
<!-- Pokemon2: https://imgsli.com/MjQ1NzM5 -->
<!-- Gundam0079: https://imgsli.com/MjQ1NzIz -->
<!-- Gundam0079 #2: https://imgsli.com/MjQ1NzMw -->
<!-- f91: https://imgsli.com/MjQ1NzMx -->
<!-- wataru: https://imgsli.com/MjQ1NzMy -->

[<img src="__assets__/visual_results/0079_visual.png" height="223px"/>](https://imgsli.com/MjQ1NzIz) [<img src="__assets__/visual_results/0079_2_visual.png" height="223px"/>](https://imgsli.com/MjQ1NzMw) 

[<img src="__assets__/visual_results/pokemon_visual.png" height="223px"/>](https://imgsli.com/MjQ1NzIy) [<img src="__assets__/visual_results/pokemon2_visual.png" height="223px"/>](https://imgsli.com/MjQ1NzM5)

[<img src="__assets__/visual_results/eva_visual.png" height="223px"/>](https://imgsli.com/MjQ1NzIx) [<img src="__assets__/visual_results/kiteret_visual.png" height="223px"/>](https://imgsli.com/MjQ1NzE0) 

[<img src="__assets__/visual_results/f91_visual.png" height="223px"/>](https://imgsli.com/MjQ1NzMx) [<img src="__assets__/visual_results/wataru_visual.png" height="223px"/>](https://imgsli.com/MjQ1NzMy)



<p align="center">
  <img src="__assets__/AVC_RealLQ_comparison.png">
</p>
<!--------------------------------------------  --------------------------------------------------->





## <a name="Update"></a>Update 🔥🔥🔥
- [x] Release Paper version implementation of APISR 
- [x] Release different upscaler factor weight (for 2x, 4x and more)
- [ ] Gradio demo (maybe online)



## <a name="installation"></a> Installation 🔧

```shell
git clone git@github.com:Kiteretsu77/APISR.git
cd APISR

# Create conda env
conda create -n APISR python=3.10
conda activate APISR

# Install Pytorch and other packages needed
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt


# To be absolutely sure that the tensorboard can execute. I recommend the following CMD from "https://github.com/pytorch/pytorch/issues/22676#issuecomment-534882021"
pip uninstall tb-nightly tensorboard tensorflow-estimator tensorflow-gpu tf-estimator-nightly
pip install tensorflow

# Install FFMPEG [Only needed for training; inference will not involve ffmpeg] (the following is for the linux system, Windows users can download ffmpeg from https://ffmpeg.org/download.html)
sudo apt install ffmpeg
```





## <a name="inference"></a> Inference ⚡⚡⚡
1. Download the model weight from [**model zoo**](docs/model_zoo.md) and **put the weight to "pretrained" folder**.
2. Then, Execute
    ```shell
    python test_code/inference.py --input_dir XXX  --weight_path XXX  --store_dir XXX
    ```
    If the weight you download is paper weight, the default argument of test_code/inference.py is capable of executing sample images from "__assets__" folder



## <a name="dataset_curation"></a> Dataset Curation 🧩
1. Our dataset curation pipeline is under **dataset_curation_pipeline** folder. You can collect your own dataset by sending videos into the pipeline and get the least compressed and the most informative images from the video sources. With a folder with video sources, you can execute the following to get a basic dataset:

    ```shell
    python dataset_curation_pipeline/collect.py --video_folder_dir XXXX --save_dir XXX
    ```

2. Once you get an image dataset with various aspect ratios and resolutions, you can run the following scripts

    Be careful to check **full_patch_source** && **degrade_hr_dataset_name** && **train_hr_dataset_name** (we will use variables in **opt.py** setting of the training)

    ```shell
    bash scripts/prepare_datasets.sh
    ```

    In order to decrease memory utilization and increase training efficiency, we pre-process all time-consuming pseudo-GT (**train_hr_dataset_name**) at the dataset preparation stage. 
    
    But in order to create a natural input for prediction-oriented compression, in every epoch, the degradation started from the uncropped GT (**full_patch_source**), and LR synthetic images are concurrently stored. The cropped HR GT dataset (**degrade_hr_dataset_name**) is fixed in the dataset preparation stage and won't be modified during training.
    



## <a name="train"></a> Train 💻

**The whole training process can be done in one RTX3090/4090!**

1. Prepare a dataset (AVC/API) which follows step 2 in [**Dataset Curation**](#dataset_curation)

2. Train: Please check **opt.py** carefully to setup parameters you want (modifying **Frequently Changed Setting** is usually enough)

    **Step1** (Net **L1** loss training): Run 
    ```shell
    python train_code/train.py 
    ```
    The model weights will be inside the folder 'saved_models' (same to checkpoints)

    **Step2** (GAN **Adversarial** Training): 
    1. Change opt['architecture'] in **opt.py** to "GRLGAN" and change **batch size** if you need. BTW, I don't think that, for personal training, it is needed to train 300K iter for GAN. I did that in order to follow the same setting as AnimeSR and VQDSR, but **100K ~ 130K** should have a decent visual result.

    2. Following previous works, GAN should start from L1 loss pre-trained network, so please carry a **pretrained_path** (the default path below should be fine)
    ```shell
    python train_code/train.py --pretrained_path saved_models/grl_best_generator.pth 
    ```

## Related Projects
1. Fast Anime SR acceleration: https://github.com/Kiteretsu77/FAST_Anime_VSR 
2. My previous paper (VCISR - WACV2024) as the baseline method: https://github.com/Kiteretsu77/VCISR-official 


## Citation
Please cite us if our work is useful for your research.
```
@article{wang2024apisr,
  title={APISR: Anime Production Inspired Real-World Anime Super-Resolution},
  author={Wang, Boyang and Yang, Fengyu and Yu, Xihang and Zhang, Chao and Zhao, Hanbin},
  journal={arXiv preprint arXiv:2403.01598},
  year={2024}
}
```

## Disclaimer
This project is released for academic use only. We disclaim responsibility for the distribution of the dataset. Users are solely liable for their actions. 
The project contributors are not legally affiliated with, nor accountable for, users' behaviors.


## License
This project is released under the [GPL 3.0 license](LICENSE).

## Contact
If you have any questions, please feel free to contact me at hikaridawn412316@gmail.com or boyangwa@umich.edu.

