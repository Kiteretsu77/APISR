# :european_castle: Model Zoo

- [For Paper weight](#for-paper-weight)
- [For Diverse Upscaler](#for-diverse-upscaler)



## For Paper Weight

| Models                                                                                                                          | Scale | Description                                  |
| ------------------------------------------------------------------------------------------------------------------------------- | :---- | :------------------------------------------- |
| [4x_APISR_GRL_GAN_generator](https://github.com/Kiteretsu77/APISR/releases/download/v0.1.0/4x_APISR_GRL_GAN_generator.pth)      | 4X    | 4X GRL model used in the paper               |


## For Diverse Upscaler

Actually, I am not that much like GRL. Though they can have the smallest param size with higher numerical results, they are not very memory efficient and the processing speed is slow for Transformer model. One more concern come from the TensorRT deployment, where Transformer architecture is hard to be adapted (needless to say for a modified version of Transformer like GRL).

Thus, for other weights, I will not train a GRL network and also real-world SR of GRL only supports 4x.


| Models                                                                                                                          | Scale | Description                                  |
| ------------------------------------------------------------------------------------------------------------------------------- | :---- | :------------------------------------------- |
| [2x_APISR_RRDB_GAN_generator](https://github.com/Kiteretsu77/APISR/releases/download/v0.1.0/2x_APISR_RRDB_GAN_generator.pth)    | 2X    | 2X upscaler by RRDB-6blocks                  |