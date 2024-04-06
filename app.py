'''
    Gradio demo (almost the same code as the one used in Huggingface space)
'''
import os, sys
import cv2
import time
import gradio as gr
import torch
import numpy as np
from torchvision.utils import save_image


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from test_code.inference import super_resolve_img
from test_code.test_utils import load_grl, load_rrdb, load_dat


def auto_download_if_needed(weight_path):
    if os.path.exists(weight_path):
        return
    
    if not os.path.exists("pretrained"):
        os.makedirs("pretrained")
    
    if weight_path == "pretrained/4x_APISR_RRDB_GAN_generator.pth":
        os.system("wget https://github.com/Kiteretsu77/APISR/releases/download/v0.2.0/4x_APISR_RRDB_GAN_generator.pth")
        os.system("mv 4x_APISR_RRDB_GAN_generator.pth pretrained")
    
    if weight_path == "pretrained/4x_APISR_GRL_GAN_generator.pth":
        os.system("wget https://github.com/Kiteretsu77/APISR/releases/download/v0.1.0/4x_APISR_GRL_GAN_generator.pth")
        os.system("mv 4x_APISR_GRL_GAN_generator.pth pretrained")
        
    if weight_path == "pretrained/2x_APISR_RRDB_GAN_generator.pth":
        os.system("wget https://github.com/Kiteretsu77/APISR/releases/download/v0.1.0/2x_APISR_RRDB_GAN_generator.pth")
        os.system("mv 2x_APISR_RRDB_GAN_generator.pth pretrained")
    
    if weight_path == "pretrained/4x_APISR_DAT_GAN_generator.pth":
        os.system("wget https://github.com/Kiteretsu77/APISR/releases/download/v0.3.0/4x_APISR_DAT_GAN_generator.pth")
        os.system("mv 4x_APISR_DAT_GAN_generator.pth pretrained")
    


def inference(img_path, model_name):
    
    try:
        weight_dtype = torch.float32
        
        # Load the model
        if model_name == "4xGRL":
            weight_path = "pretrained/4x_APISR_GRL_GAN_generator.pth"
            auto_download_if_needed(weight_path)
            generator = load_grl(weight_path, scale=4)  # Directly use default way now
            
        elif model_name == "4xRRDB":
            weight_path = "pretrained/4x_APISR_RRDB_GAN_generator.pth"
            auto_download_if_needed(weight_path)
            generator = load_rrdb(weight_path, scale=4)  # Directly use default way now
            
        elif model_name == "2xRRDB":
            weight_path = "pretrained/2x_APISR_RRDB_GAN_generator.pth"
            auto_download_if_needed(weight_path)
            generator = load_rrdb(weight_path, scale=2) # Directly use default way now
            
        elif model_name == "4xDAT":
            weight_path = "pretrained/4x_APISR_DAT_GAN_generator.pth"
            auto_download_if_needed(weight_path)
            generator = load_dat(weight_path, scale=4) # Directly use default way now
            
        else:
            raise gr.Error("We don't support such Model")
        
        generator = generator.to(dtype=weight_dtype)


        # In default, we will automatically use crop to match 4x size
        super_resolved_img = super_resolve_img(generator, img_path, output_path=None, weight_dtype=weight_dtype, downsample_threshold=720, crop_for_4x=True)
        store_name = str(time.time()) + ".png"
        save_image(super_resolved_img, store_name)
        outputs = cv2.imread(store_name)
        outputs = cv2.cvtColor(outputs, cv2.COLOR_RGB2BGR)
        os.remove(store_name)
        
        return outputs
    
    
    except Exception as error:
        raise gr.Error(f"global exception: {error}")



if __name__ == '__main__':
    
    MARKDOWN = \
    """
    ## <p style='text-align: center'> APISR: Anime Production Inspired Real-World Anime Super-Resolution (CVPR 2024) </p>
    
    [GitHub](https://github.com/Kiteretsu77/APISR) | [Paper](https://arxiv.org/abs/2403.01598)
    APISR aims at restoring and enhancing low-quality low-resolution **anime** images and video sources with various degradations from real-world scenarios.
    
    ### Note: Due to memory restriction, all images whose short side is over 720 pixel will be downsampled to 720 pixel with the same aspect ratio.  E.g., 1920x1080 -> 1280x720
    ### Note: Please check [Model Zoo](https://github.com/Kiteretsu77/APISR/blob/main/docs/model_zoo.md) for the description of each weight.
    
    If APISR is helpful, please help star the [GitHub Repo](https://github.com/Kiteretsu77/APISR). Thanks! 
    """

    block = gr.Blocks().queue(max_size=10)
    with block:
        with gr.Row():
            gr.Markdown(MARKDOWN)
        with gr.Row(elem_classes=["container"]):
            with gr.Column(scale=2):
                input_image = gr.Image(type="filepath", label="Input")
                model_name = gr.Dropdown(
                    [
                        "2xRRDB",
                        "4xRRDB",
                        "4xGRL",
                        "4xDAT",
                    ],
                    type="value",
                    value="4xGRL",
                    label="model",
                )
                run_btn = gr.Button(value="Submit")

            with gr.Column(scale=3):
                output_image = gr.Image(type="numpy", label="Output image")

        with gr.Row(elem_classes=["container"]):
            gr.Examples(
                [
                    ["__assets__/lr_inputs/image-00277.png"],
                    ["__assets__/lr_inputs/image-00542.png"],
                    ["__assets__/lr_inputs/41.png"],
                    ["__assets__/lr_inputs/f91.jpg"],
                    ["__assets__/lr_inputs/image-00440.png"],
                    ["__assets__/lr_inputs/image-00164.jpg"],
                    ["__assets__/lr_inputs/img_eva.jpeg"],
                    ["__assets__/lr_inputs/naruto.jpg"],
                ],
                [input_image],
            )

        run_btn.click(inference, inputs=[input_image, model_name], outputs=[output_image])

    block.launch()
