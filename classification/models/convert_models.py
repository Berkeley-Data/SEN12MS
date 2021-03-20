#!/usr/bin/env python

import torch
import argparse
from torchvision import models

# Command: python convert_models.py -i <path to the moco model xyz.pth> -bb True
# -bb True indicates to extract backbone weights. -bb False indicates to extract encoder query weights

if __name__ == "__main__":

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    parser = argparse.ArgumentParser(
        description='This script extracts the backbone from a OpenSelfSup moco model.')

    parser.add_argument('-i', '--inputmodel', type=str,
                        help="Input model file name")

    parser.add_argument('-bb', '--backbone', default=True, type=str2bool,
                        help="whether to extract backbone or encoder keys?")

    args = parser.parse_args()

    obj = torch.load(args.inputmodel, map_location="cpu")

    # Note: Original resnet50 model will not have any state_dict key. All the key values are directly under
    # the main dictionary

    #Moco resnet model will have the following keys
    # 1) meta  - This is the whole .py config file
    # 2) state_dict
    # 3) optimizer
    obj = obj["state_dict"]

    newmodel = {}
    for k, v in obj.items():
        #print(k,':::',v)
        #print(k)
       # continue;

        # Added verbose checks for easy understanding on what we are ignoring
        # There won't be any performance issue as it's a one time run during conversion
        if k.startswith("queue"):
            continue
        elif k.startswith("input_module"):
            continue
        elif k.startswith("encoder_k"):
            continue
        elif k.startswith("encoder_q.1.mlp"):
            continue
        elif k.startswith("encoder_k.0."):
            continue
        elif k.startswith("encoder_k.1.mlp"):
            continue
        elif k.startswith("backbone.") or k.startswith("encoder_q.0."):
            if not k.endswith("num_batches_tracked"):
                if args.backbone == True and k.startswith("backbone."):
                    # Extract the backbone
                    k = k.replace("backbone.", "")
                    newmodel[k] = v
                elif args.backbone == False and k.startswith("encoder_q.0."):
                    # Extract the query encoder
                    k = k.replace("encoder_q.0.", "")
                    newmodel[k] = v

    # Moco model doesn't have fc.weight and fc.bias keys. But, these are required by ResNet. So, load from ResNet
    # and save them to the extracted model. Otherwise the saved model can't be loaded with model.load_state_dict() API
    # and results in missing keys
    resnet = models.resnet50(pretrained=False)
    newmodel["fc.weight"] = resnet.fc.weight
    newmodel["fc.bias"] = resnet.fc.bias

    res = {
        "state_dict": newmodel,
        "__author__": "OpenSelfSup",
        "matching_heuristics": True
    }

    output_file_name = args.inputmodel.split('.')[0]

    if args.backbone == True:
        output_file_name = output_file_name+'_backbone.pth'
    else:
        output_file_name = output_file_name + '_queryencoder.pth'

    with open(output_file_name, "wb") as f:
        torch.save(res, f)

    # Test the model by loading it
    resnet = models.resnet50(pretrained=False)
    resnet.load_state_dict(res["state_dict"])


