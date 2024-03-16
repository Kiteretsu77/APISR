import os, sys
import cv2
import gradio as gr
import torch
import numpy as np
from torchvision.utils import save_image


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from test_code.inference import super_resolve_img
from test_code.test_utils import load_grl, load_rrdb


def auto_download_if_needed(weight_path):
    if os.path.exists(weight_path):
        return
    
    if not os.path.exists("pretrained"):
        os.makedirs("pretrained")
    
    if weight_path == "pretrained/4x_APISR_GRL_GAN_generator.pth":
        os.system("wget https://github.com/Kiteretsu77/APISR/releases/download/v0.1.0/4x_APISR_GRL_GAN_generator.pth")
        os.system("mv 4x_APISR_GRL_GAN_generator.pth pretrained")
        
    if weight_path == "pretrained/2x_APISR_RRDB_GAN_generator.pth":
        os.system("wget https://github.com/Kiteretsu77/APISR/releases/download/v0.1.0/2x_APISR_RRDB_GAN_generator.pth")
        os.system("mv 2x_APISR_RRDB_GAN_generator.pth pretrained")



def inference(img_path, model_name):
    
    try:
        weight_dtype = torch.float32
        
        # Load the model
        if model_name == "4xGRL":
            weight_path = "pretrained/4x_APISR_GRL_GAN_generator.pth"
            auto_download_if_needed(weight_path)
            generator = load_grl(weight_path, scale=4)  # Directly use default way now
            
        elif model_name == "2xRRDB":
            weight_path = "pretrained/2x_APISR_RRDB_GAN_generator.pth"
            auto_download_if_needed(weight_path)
            generator = load_rrdb(weight_path, scale=2) # Directly use default way now
            
        else:
            raise gr.Error(error)
        
        generator = generator.to(dtype=weight_dtype)


        # In default, we will automatically use crop to match 4x size
        super_resolved_img = super_resolve_img(generator, img_path, output_path=None, weight_dtype=weight_dtype, crop_for_4x=True)
        save_image(super_resolved_img, "SR_result.png")
        outputs = cv2.imread("SR_result.png")
        outputs = cv2.cvtColor(outputs, cv2.COLOR_RGB2BGR)
        
        return outputs
    
    
    except Exception as error:
        raise gr.Error(f"global exception: {error}")



if __name__ == '__main__':
    
    MARKDOWN = \
    """
    ## APISR: Anime Production Inspired Real-World Anime Super-Resolution (CVPR 2024)

    [GitHub](https://github.com/Kiteretsu77/APISR) | [Paper](https://arxiv.org/abs/2403.01598)

    If APISR is helpful for you, please help star the GitHub Repo. Thanks!
    """

    block = gr.Blocks().queue()
    with block:
        with gr.Row():
            gr.Markdown(MARKDOWN)
        with gr.Row(elem_classes=["container"]):
            with gr.Column(scale=2):
                input_image = gr.Image(type="filepath", label="Input")
                model_name = gr.Dropdown(
                    [
                        "2xRRDB",
                        "4xGRL"
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
                    ["__assets__/lr_inputs/image-00164.png"],
                    ["__assets__/lr_inputs/img_eva.jpeg"],
                ],
                [input_image],
            )

        run_btn.click(inference, inputs=[input_image, model_name], outputs=[output_image])

    block.launch()