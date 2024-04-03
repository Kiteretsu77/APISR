'''
    Clean unnecessary information in the weight (*.pth)
'''
import torch


if __name__ == "__main__":
    weight_path = "saved_models/4x_DATGAN_small/datgan_closest_generator.pth"
    store_path = "4x_APISR_DAT_GAN_generator.pth"

    # Load the checkpoint
    checkpoint_g = torch.load(weight_path)
    keys = []
    for key in checkpoint_g: 
        keys.append(key)
        print(key)
    for key in keys:
        if key != "model_state_dict":
            del checkpoint_g[key]
        

    # Access the weight
    old_keys = [key for key in checkpoint_g['model_state_dict']]
    for old_key in old_keys:
        if old_key[:10] == "_orig_mod.":
            new_key = old_key[10:]
            checkpoint_g['model_state_dict'][new_key] = checkpoint_g['model_state_dict'][old_key]
            del checkpoint_g['model_state_dict'][old_key]

    torch.save(checkpoint_g, store_path)




