import os, sys
from collections import OrderedDict
import cv2
import torch.nn as nn
import torch
from torchvision import models
import torchvision.transforms as transforms

'''
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 112, 112]           9,408
       BatchNorm2d-2         [-1, 64, 112, 112]             128
              ReLU-3         [-1, 64, 112, 112]               0
         MaxPool2d-4           [-1, 64, 56, 56]               0
            Conv2d-5           [-1, 64, 56, 56]           4,096
       BatchNorm2d-6           [-1, 64, 56, 56]             128
              ReLU-7           [-1, 64, 56, 56]               0
            Conv2d-8           [-1, 64, 56, 56]          36,864
       BatchNorm2d-9           [-1, 64, 56, 56]             128
             ReLU-10           [-1, 64, 56, 56]               0
           Conv2d-11          [-1, 256, 56, 56]          16,384
      BatchNorm2d-12          [-1, 256, 56, 56]             512
           Conv2d-13          [-1, 256, 56, 56]          16,384
      BatchNorm2d-14          [-1, 256, 56, 56]             512
             ReLU-15          [-1, 256, 56, 56]               0
       Bottleneck-16          [-1, 256, 56, 56]               0
           Conv2d-17           [-1, 64, 56, 56]          16,384
      BatchNorm2d-18           [-1, 64, 56, 56]             128
             ReLU-19           [-1, 64, 56, 56]               0
           Conv2d-20           [-1, 64, 56, 56]          36,864
      BatchNorm2d-21           [-1, 64, 56, 56]             128
             ReLU-22           [-1, 64, 56, 56]               0
           Conv2d-23          [-1, 256, 56, 56]          16,384
      BatchNorm2d-24          [-1, 256, 56, 56]             512
             ReLU-25          [-1, 256, 56, 56]               0
       Bottleneck-26          [-1, 256, 56, 56]               0
           Conv2d-27           [-1, 64, 56, 56]          16,384
      BatchNorm2d-28           [-1, 64, 56, 56]             128
             ReLU-29           [-1, 64, 56, 56]               0
           Conv2d-30           [-1, 64, 56, 56]          36,864
      BatchNorm2d-31           [-1, 64, 56, 56]             128
             ReLU-32           [-1, 64, 56, 56]               0
           Conv2d-33          [-1, 256, 56, 56]          16,384
      BatchNorm2d-34          [-1, 256, 56, 56]             512
             ReLU-35          [-1, 256, 56, 56]               0
       Bottleneck-36          [-1, 256, 56, 56]               0
           Conv2d-37          [-1, 128, 56, 56]          32,768
      BatchNorm2d-38          [-1, 128, 56, 56]             256
             ReLU-39          [-1, 128, 56, 56]               0
           Conv2d-40          [-1, 128, 28, 28]         147,456
      BatchNorm2d-41          [-1, 128, 28, 28]             256
             ReLU-42          [-1, 128, 28, 28]               0
           Conv2d-43          [-1, 512, 28, 28]          65,536
      BatchNorm2d-44          [-1, 512, 28, 28]           1,024
           Conv2d-45          [-1, 512, 28, 28]         131,072
      BatchNorm2d-46          [-1, 512, 28, 28]           1,024
             ReLU-47          [-1, 512, 28, 28]               0
       Bottleneck-48          [-1, 512, 28, 28]               0
           Conv2d-49          [-1, 128, 28, 28]          65,536
      BatchNorm2d-50          [-1, 128, 28, 28]             256
             ReLU-51          [-1, 128, 28, 28]               0
           Conv2d-52          [-1, 128, 28, 28]         147,456
      BatchNorm2d-53          [-1, 128, 28, 28]             256
             ReLU-54          [-1, 128, 28, 28]               0
           Conv2d-55          [-1, 512, 28, 28]          65,536
      BatchNorm2d-56          [-1, 512, 28, 28]           1,024
             ReLU-57          [-1, 512, 28, 28]               0
       Bottleneck-58          [-1, 512, 28, 28]               0
           Conv2d-59          [-1, 128, 28, 28]          65,536
      BatchNorm2d-60          [-1, 128, 28, 28]             256
             ReLU-61          [-1, 128, 28, 28]               0
           Conv2d-62          [-1, 128, 28, 28]         147,456
      BatchNorm2d-63          [-1, 128, 28, 28]             256
             ReLU-64          [-1, 128, 28, 28]               0
           Conv2d-65          [-1, 512, 28, 28]          65,536
      BatchNorm2d-66          [-1, 512, 28, 28]           1,024
             ReLU-67          [-1, 512, 28, 28]               0
       Bottleneck-68          [-1, 512, 28, 28]               0
           Conv2d-69          [-1, 128, 28, 28]          65,536
      BatchNorm2d-70          [-1, 128, 28, 28]             256
             ReLU-71          [-1, 128, 28, 28]               0
           Conv2d-72          [-1, 128, 28, 28]         147,456
      BatchNorm2d-73          [-1, 128, 28, 28]             256
             ReLU-74          [-1, 128, 28, 28]               0
           Conv2d-75          [-1, 512, 28, 28]          65,536
      BatchNorm2d-76          [-1, 512, 28, 28]           1,024
             ReLU-77          [-1, 512, 28, 28]               0
       Bottleneck-78          [-1, 512, 28, 28]               0
           Conv2d-79          [-1, 256, 28, 28]         131,072
      BatchNorm2d-80          [-1, 256, 28, 28]             512
             ReLU-81          [-1, 256, 28, 28]               0
           Conv2d-82          [-1, 256, 14, 14]         589,824
      BatchNorm2d-83          [-1, 256, 14, 14]             512
             ReLU-84          [-1, 256, 14, 14]               0
           Conv2d-85         [-1, 1024, 14, 14]         262,144
      BatchNorm2d-86         [-1, 1024, 14, 14]           2,048
           Conv2d-87         [-1, 1024, 14, 14]         524,288
      BatchNorm2d-88         [-1, 1024, 14, 14]           2,048
             ReLU-89         [-1, 1024, 14, 14]               0
       Bottleneck-90         [-1, 1024, 14, 14]               0
           Conv2d-91          [-1, 256, 14, 14]         262,144
      BatchNorm2d-92          [-1, 256, 14, 14]             512
             ReLU-93          [-1, 256, 14, 14]               0
           Conv2d-94          [-1, 256, 14, 14]         589,824
      BatchNorm2d-95          [-1, 256, 14, 14]             512
             ReLU-96          [-1, 256, 14, 14]               0
           Conv2d-97         [-1, 1024, 14, 14]         262,144
      BatchNorm2d-98         [-1, 1024, 14, 14]           2,048
             ReLU-99         [-1, 1024, 14, 14]               0
      Bottleneck-100         [-1, 1024, 14, 14]               0
          Conv2d-101          [-1, 256, 14, 14]         262,144
     BatchNorm2d-102          [-1, 256, 14, 14]             512
            ReLU-103          [-1, 256, 14, 14]               0
          Conv2d-104          [-1, 256, 14, 14]         589,824
     BatchNorm2d-105          [-1, 256, 14, 14]             512
            ReLU-106          [-1, 256, 14, 14]               0
          Conv2d-107         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-108         [-1, 1024, 14, 14]           2,048
            ReLU-109         [-1, 1024, 14, 14]               0
      Bottleneck-110         [-1, 1024, 14, 14]               0
          Conv2d-111          [-1, 256, 14, 14]         262,144
     BatchNorm2d-112          [-1, 256, 14, 14]             512
            ReLU-113          [-1, 256, 14, 14]               0
          Conv2d-114          [-1, 256, 14, 14]         589,824
     BatchNorm2d-115          [-1, 256, 14, 14]             512
            ReLU-116          [-1, 256, 14, 14]               0
          Conv2d-117         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-118         [-1, 1024, 14, 14]           2,048
            ReLU-119         [-1, 1024, 14, 14]               0
      Bottleneck-120         [-1, 1024, 14, 14]               0
          Conv2d-121          [-1, 256, 14, 14]         262,144
     BatchNorm2d-122          [-1, 256, 14, 14]             512
            ReLU-123          [-1, 256, 14, 14]               0
          Conv2d-124          [-1, 256, 14, 14]         589,824
     BatchNorm2d-125          [-1, 256, 14, 14]             512
            ReLU-126          [-1, 256, 14, 14]               0
          Conv2d-127         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-128         [-1, 1024, 14, 14]           2,048
            ReLU-129         [-1, 1024, 14, 14]               0
      Bottleneck-130         [-1, 1024, 14, 14]               0
          Conv2d-131          [-1, 256, 14, 14]         262,144
     BatchNorm2d-132          [-1, 256, 14, 14]             512
            ReLU-133          [-1, 256, 14, 14]               0
          Conv2d-134          [-1, 256, 14, 14]         589,824
     BatchNorm2d-135          [-1, 256, 14, 14]             512
            ReLU-136          [-1, 256, 14, 14]               0
          Conv2d-137         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-138         [-1, 1024, 14, 14]           2,048
            ReLU-139         [-1, 1024, 14, 14]               0
      Bottleneck-140         [-1, 1024, 14, 14]               0
          Conv2d-141          [-1, 512, 14, 14]         524,288
     BatchNorm2d-142          [-1, 512, 14, 14]           1,024
            ReLU-143          [-1, 512, 14, 14]               0
          Conv2d-144            [-1, 512, 7, 7]       2,359,296
     BatchNorm2d-145            [-1, 512, 7, 7]           1,024
            ReLU-146            [-1, 512, 7, 7]               0
          Conv2d-147           [-1, 2048, 7, 7]       1,048,576
     BatchNorm2d-148           [-1, 2048, 7, 7]           4,096
          Conv2d-149           [-1, 2048, 7, 7]       2,097,152
     BatchNorm2d-150           [-1, 2048, 7, 7]           4,096
            ReLU-151           [-1, 2048, 7, 7]               0
      Bottleneck-152           [-1, 2048, 7, 7]               0
          Conv2d-153            [-1, 512, 7, 7]       1,048,576
     BatchNorm2d-154            [-1, 512, 7, 7]           1,024
            ReLU-155            [-1, 512, 7, 7]               0
          Conv2d-156            [-1, 512, 7, 7]       2,359,296
     BatchNorm2d-157            [-1, 512, 7, 7]           1,024
            ReLU-158            [-1, 512, 7, 7]               0
          Conv2d-159           [-1, 2048, 7, 7]       1,048,576
     BatchNorm2d-160           [-1, 2048, 7, 7]           4,096
            ReLU-161           [-1, 2048, 7, 7]               0
      Bottleneck-162           [-1, 2048, 7, 7]               0
          Conv2d-163            [-1, 512, 7, 7]       1,048,576
     BatchNorm2d-164            [-1, 512, 7, 7]           1,024
            ReLU-165            [-1, 512, 7, 7]               0
          Conv2d-166            [-1, 512, 7, 7]       2,359,296
     BatchNorm2d-167            [-1, 512, 7, 7]           1,024
            ReLU-168            [-1, 512, 7, 7]               0
          Conv2d-169           [-1, 2048, 7, 7]       1,048,576
     BatchNorm2d-170           [-1, 2048, 7, 7]           4,096
            ReLU-171           [-1, 2048, 7, 7]               0
      Bottleneck-172           [-1, 2048, 7, 7]               0
AdaptiveMaxPool2d-173          [-1, 2048, 1, 1]               0
AdaptiveAvgPool2d-174          [-1, 2048, 1, 1]               0
AdaptiveConcatPool2d-175       [-1, 4096, 1, 1]               0
         Flatten-176                 [-1, 4096]               0
     BatchNorm1d-177                 [-1, 4096]           8,192
         Dropout-178                 [-1, 4096]               0
          Linear-179                  [-1, 512]       2,097,664
            ReLU-180                  [-1, 512]               0
     BatchNorm1d-181                  [-1, 512]           1,024
         Dropout-182                  [-1, 512]               0
          Linear-183                 [-1, 6000]       3,078,000
================================================================
Total params: 28,692,912
Trainable params: 28,692,912
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 286.75
Params size (MB): 109.45
Estimated Total Size (MB): 396.78
----------------------------------------------------------------
'''


class AdaptiveConcatPool2d(nn.Module):
    """
    Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`.
    Source: Fastai. This code was taken from the fastai library at url
    https://github.com/fastai/fastai/blob/master/fastai/layers.py#L176
    """
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)
    

class Flatten(nn.Module):
    """
    Flatten `x` to a single dimension. Adapted from fastai's Flatten() layer,
    at https://github.com/fastai/fastai/blob/master/fastai/layers.py#L25
    """
    def __init__(self): super().__init__()
    def forward(self, x): return x.view(x.size(0), -1)


def bn_drop_lin(n_in:int, n_out:int, bn:bool=True, p:float=0., actn=None):
    """
    Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`.
    Adapted from Fastai at https://github.com/fastai/fastai/blob/master/fastai/layers.py#L44
    """
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers

def create_head(top_n_tags, nf, ps=0.5):
    nc = top_n_tags
    
    lin_ftrs = [nf, 512, nc]
    p1 = 0.25 # dropout for second last layer
    p2 = 0.5 # dropout for last layer

    actns = [nn.ReLU(inplace=True),] + [None]
    pool = AdaptiveConcatPool2d()
    layers = [pool, Flatten()]
    
    layers += [
        *bn_drop_lin(lin_ftrs[0], lin_ftrs[1], True, p1, nn.ReLU(inplace=True)),
        *bn_drop_lin(lin_ftrs[1], lin_ftrs[2], True, p2)
    ]
    
    return nn.Sequential(*layers)


def _resnet(base_arch, top_n, **kwargs):
    cut = -2
    s = base_arch(pretrained=False, **kwargs)
    body = nn.Sequential(*list(s.children())[:cut])

    if base_arch in [models.resnet18, models.resnet34]:
        num_features_model = 512
    elif base_arch in [models.resnet50, models.resnet101]:
        num_features_model = 2048

    nf = num_features_model * 2
    nc = top_n

    # head = create_head(nc, nf)
    model = body # nn.Sequential(body, head)

    return model


def resnet50(pretrained=True, progress=True, top_n=6000, **kwargs):
    r""" 
    Resnet50 model trained on the full Danbooru2018 dataset's top 6000 tags

    Args:
        pretrained (bool): kwargs, load pretrained weights into the model.
        top_n (int): kwargs, pick to load the model for predicting the top `n` tags,
            currently only supports top_n=6000.
    """
    model = _resnet(models.resnet50, top_n, **kwargs)       # Take Resnet without the head (we don't care about final FC layers)

    if pretrained:
        if top_n == 6000: 
            state = torch.hub.load_state_dict_from_url("https://github.com/RF5/danbooru-pretrained/releases/download/v0.1/resnet50-13306192.pth", 
                                                   progress=progress)
            old_keys = [key for key in state]
            for old_key in old_keys:
                if old_key[0] == '0':
                    new_key = old_key[2:]
                    state[new_key] = state[old_key]
                    del state[old_key]
                elif old_key[0] == '1':
                    del state[old_key]
                
            model.load_state_dict(state)
        else:
            raise ValueError("Sorry, the resnet50 model only supports the top-6000 tags \
                at the moment")

    
    return model




class resnet50_Extractor(nn.Module):
    """ResNet50 network for feature extraction.
    """
    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    
    def __init__(self,
                 model,
                 layer_labels,
                 use_input_norm=True,
                 range_norm=False,
                 requires_grad=False
                 ):
        super(resnet50_Extractor, self).__init__()


        self.model = model
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm
        self.layer_labels = layer_labels
        self.activation = {}


        # Extract needed features
        for layer_label in layer_labels:
            elements = layer_label.split('_')
            if len(elements) == 1:
                # modified_net[layer_label] = getattr(model, elements[0])
                getattr(self.model, elements[0]).register_forward_hook(self.get_activation(layer_label))
            else:
                body_layer = self.model
                for element in elements[:-1]:
                    # Iterate until the last element
                    assert(isinstance(int(element), int))
                    body_layer = body_layer[int(element)]
                getattr(body_layer, elements[-1]).register_forward_hook(self.get_activation(layer_label))


        # Set as evaluation
        if not requires_grad:
            self.model.eval()
            for param in self.parameters():
                param.requires_grad = False


        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [0, 1]
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))



    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        if self.range_norm:
            x = (x + 1) / 2
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        # Execute model first
        output = self.model(x) # Zomby input
        
        # Extract the layers we need
        store = {}
        for layer_label in self.layer_labels:
            store[layer_label] = self.activation[layer_label]


        return store


class Anime_PerceptualLoss(nn.Module):
    """Anime Perceptual loss

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 perceptual_weight=1.0,
                 criterion='l1'):
        super(Anime_PerceptualLoss, self).__init__()

    
        model = resnet50()
        self.perceptual_weight = perceptual_weight
        self.layer_weights = layer_weights
        self.layer_labels = layer_weights.keys()
        self.resnet50 = resnet50_Extractor(model, self.layer_labels).cuda()

        if criterion == 'l1':
            self.criterion = torch.nn.L1Loss()
        else:
            raise NotImplementedError("We don't support such criterion loss in perceptual loss")


    def forward(self, gen, gt):
        """Forward function.

        Args:
            gen (Tensor):   Input tensor with shape (n, c, h, w).
            gt (Tensor):    Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        gen_features = self.resnet50(gen)
        gt_features = self.resnet50(gt.detach())
        

        temp_store = []

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for idx, k in enumerate(gen_features.keys()):
                raw_comparison = self.criterion(gen_features[k], gt_features[k])
                percep_loss += raw_comparison * self.layer_weights[k]

                # print("layer" + str(idx) + " has loss " + str(raw_comparison.cpu().numpy()))
                # temp_store.append(float(raw_comparison.cpu().numpy()))

            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # 第一个是为了Debug purpose
        if len(temp_store) != 0:
            return temp_store, percep_loss
        else:
            return percep_loss




if __name__ == "__main__":
    import torchvision.transforms as transforms
    import cv2
    import collections


    loss = Anime_PerceptualLoss({"0": 0.5, "4_2_conv3": 20, "5_3_conv3": 30, "6_5_conv3": 1, "7_2_conv3": 1}).cuda()

    
    store = collections.defaultdict(list)
    for img_name in sorted(os.listdir('datasets/train_gen/')):
        gen = transforms.ToTensor()(cv2.imread('datasets/train_gen/'+img_name)).cuda()
        gt = transforms.ToTensor()(cv2.imread('datasets/train_hr_anime_usm/'+img_name)).cuda()
        temp_store, _ = loss(gen, gt)

        for idx in range(len(temp_store)):
            store[idx].append(temp_store[idx])

    for idx in range(len(store)):
        print("Average layer" + str(idx) + " has loss " + str(sum(store[idx]) / len(store[idx])))


    # model = loss.vgg
    # pytorch_total_params = sum(p.numel() for p in model.parameters())
    # print(f"Perceptual VGG has param {pytorch_total_params//1000000} M params")