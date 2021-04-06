#!/usr/bin/env python

import torch
import argparse
from torchvision import models
import wandb
import os
import torch.nn as nn

# Command: python convert_moco_to_resnet50.py -i <path to the moco model xyz.pth> -bb True
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
        description='This script extracts/converts the backbone from a OpenSelfSup moco model')

    parser.add_argument('-n', '--n_inputs', type=int, default=3,
                        help="number of input channel for conv1")
    parser.add_argument('-o', '--outputdir', type=str,
                        help="output directory to save the model")
    parser.add_argument('-i', '--inputmodel', type=str,
                        help="W&B run id or local path (e.g., 3l4yg63k)")
    parser.add_argument('-bb', '--backbone', default=True, type=str2bool,
                        help="whether to extracts/converts backbone or extracts encoder keys?")

    args = parser.parse_args()

    checkpoint = args.inputmodel
    if os.path.exists(checkpoint):
        obj = torch.load(checkpoint, map_location=torch.device('cpu'))
        output_file_name = checkpoint.split('.')[0]
    else:
        # wandb.init(project=checkpoint.split("/")[0])
        # if not checkpoint is not valid path, check for wandb
        run_id = checkpoint.split("/")[1]
        tmpdir = os.path.join(args.outputdir, run_id)
        os.mkdir(tmpdir)
        restored_model = wandb.restore(f'latest.pth', run_path=f"{checkpoint}", root=tmpdir, replace=False)
        if restored_model is None:
            raise Exception(f"failed to load the model from runid or path: {checkpoint} ")
        obj = torch.load(restored_model.name, map_location=torch.device('cpu'))
        output_file_name = os.path.join(args.outputdir, run_id)

    # Note: Original resnet50 model will not have any state_dict key. All the key values are directly under
    # the main dictionary

    #Moco resnet model will have the following keys
    # 1) meta  - This is the whole .py config file
    # 2) state_dict
    # 3) optimizer
    obj = obj["state_dict"]

    newmodel = {}
    inputmodule_model = {}
    for k, v in obj.items():
        #print(k,':::',v)
        #print(k)
       # continue;

        # Added verbose checks for easy understanding on what we are ignoring
        # There won't be any performance issue as it's a one time run during conversion
        if k.startswith("queue"):
            continue
        elif k.startswith("input_module_k"):
            continue
        elif k.startswith("input_module_q"):
            if not k.endswith("num_batches_tracked"):
                # Input module is added under a separate key
                # Note: Input module values are available only for query and key encoders in moco model.
                k = k.replace("input_module_q", "input_module")
                inputmodule_model[k] = v
                #print("K: ",k, " V:",v)
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
        "input_module": inputmodule_model,
        "__author__": "OpenSelfSup",
        "matching_heuristics": True
    }

    if args.backbone == True:
        output_file_name = output_file_name+'_bb_converted.pth'
    else:
        output_file_name = output_file_name + '_qe_converted.pth'

    with open(output_file_name, "wb") as f:
        torch.save(res, f)

    # Test the model by loading it
    resnet = models.resnet50(pretrained=False)
    # [todo] taeil: input channel should be configurable depending on which moco module we are transferring.
    resnet.conv1 = nn.Conv2d(args.n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resnet.load_state_dict(res["state_dict"])

    print(f"{args.n_inputs} input channels for conv1 is complete and save at {output_file_name}")

