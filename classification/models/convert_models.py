#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import sys
import torch
from torchvision import models

if __name__ == "__main__":
    input = sys.argv[1]
    obj = torch.load(input, map_location="cpu")

    if hasattr(obj, "state_dict"):
        obj = obj.state_dict()
    elif "state_dict" in obj:
        obj = obj["state_dict"]

    newmodel = {}
    for k, v in obj.items():
        if k[:7] == "module.":
            k=k[7:]
        old_k = k
        if "layer" not in k:
            k = "stem." + k
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")

        k = k.replace('stem.fpn', 'backbone.fpn')

        print(old_k, "->", k)
        newmodel[k] = v.detach().numpy()

    res = {
        # [todo] the name of state_dict should match the one
        "state_dict": newmodel,
        "__author__": "OpenSelfSup",
        "matching_heuristics": True
    }

    resnet = models.resnet50(pretrained=False)
    resnet.load_state_dict(newmodel)

    assert sys.argv[2].endswith('.pth')
    with open(sys.argv[2], "wb") as f:
        torch.save(res, f)