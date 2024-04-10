# :european_castle: Model Zoo

- [Paper weight](#for-paper-weight)
- [Diverse Upscaler Architecture](#for-diverse-upscaler)


Visual Comparisons between models can be found [here](https://imgsli.com/MjU0MjI0). I would say that, currently, I prefer 4XDAT model (but more computational intensive).

## Paper Weight
The time we calculated is based on our device (3090Ti) with input 256x256 (without half precision or any other acceleration). This is only a reference, so it's better to focus on the **relative process time difference** between different models and different scaling factors.

| Models                                                                                                                          | Scale | Param |  Time | Description                                  |
| ------------------------------------------------------------------------------------------------------------------------------- | :---- | :-----| :-----  | :------------------------------------------- |
| [4x_APISR_GRL_GAN_generator](https://github.com/Kiteretsu77/APISR/releases/download/v0.1.0/4x_APISR_GRL_GAN_generator.pth)      | 4X    | 1.03 M | 0.078s | 4X GRL model used in the paper               |




## Diverse Upscaler Architecture

Actually, I am not that much like GRL. Though they can have the smallest param size with higher numerical results, they are not very memory efficient and the processing speed is slow. Moreover, they only support 4x upscaling factor for the real-world SR part.

My main target will be **2x** and **4x**. The network structure will be chosen from the following perspective: **(1)** A Larger Transformer-based model (e.g., DAT, HAT) for better representation learning; **(2)** Popular models (e.g., RRDB) that are already deployed everywhere to decrease the code needed for deployment; **(3)** An even smaller model for fast inference (this probably needs a while for selection).


| Models                                                                                                                          | Scale | Param | Time | Description                                  |
| ------------------------------------------------------------------------------------------------------------------------------- | :---- | :----- | :---- | :------------------------------------------- |
| [2x_APISR_RRDB_GAN_generator](https://github.com/Kiteretsu77/APISR/releases/download/v0.1.0/2x_APISR_RRDB_GAN_generator.pth)    | 2X    | 4.47 M | 0.010s | 2X upscaler by RRDB-6blocks                  |
| [4x_APISR_RRDB_GAN_generator](https://github.com/Kiteretsu77/APISR/releases/download/v0.2.0/4x_APISR_RRDB_GAN_generator.pth)    | 4X    | 4.47 M | 0.035s | 4X upscaler by RRDB-6blocks (Probably needs to tune twin perceptual loss hyperparameter to decrease unwanted color artifacts)                 |
| [4x_APISR_DAT_GAN_generator](https://github.com/Kiteretsu77/APISR/releases/download/v0.3.0/4x_APISR_DAT_GAN_generator.pth)      | 4X    | 10.89M | 0.683s | 4X upscaler by DAT-Small                     |
