# Modified from Jian Kang, https://www.rsim.tu-berlin.de/menue/team/dring_jian_kang/
# Modified by Yu-Lun Wu, TUM
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function
from torchvision import models


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data)

def fc_init_weights(m):
    if type(m) == nn.Linear:
        init.kaiming_normal_(m.weight.data)


class ResNet18(nn.Module):
    def __init__(self, n_inputs = 12, numCls = 17):
        super().__init__()

        resnet = models.resnet18(pretrained=False)
        
        self.conv1 = nn.Conv2d(n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.FC = nn.Linear(512, numCls)

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x):

        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        logits = self.FC(x)

        return logits




class ResNet34(nn.Module):
    def __init__(self, n_inputs = 12, numCls = 17):
        super().__init__()

        resnet = models.resnet34(pretrained=False)

        self.conv1 = nn.Conv2d(n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.FC = nn.Linear(512, numCls)
        
        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x):

        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        logits = self.FC(x)

        return logits



class ResNet50(nn.Module):
    def __init__(self, n_inputs = 12, numCls = 17):
        super().__init__()

        resnet = models.resnet50(pretrained=False)

        self.conv1 = nn.Conv2d(n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.FC = nn.Linear(2048, numCls)

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        logits = self.FC(x)

        return logits

class ResNet50_1x1(nn.Module):
    def __init__(self, n_inputs = 12, numCls = 17):
        super().__init__()

        resnet = models.resnet50(pretrained=False)

        self.Conv1x1Block = nn.Sequential(
            nn.Conv2d(n_inputs, 3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=(256, 256), stride=(2, 2), padding=(3, 3), bias=False)
        # self.conv1 = nn.Conv2d(n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = nn.Sequential(
            self.Conv1x1Block,
            self.conv1, # self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )

        self.FC = nn.Linear(2048, numCls)

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        logits = self.FC(x)

        return logits


class Moco_1x1(nn.Module):
    def __init__(self, mocoModel, n_inputs = 12, numCls = 17):
        super().__init__()

        resnet = models.resnet50(pretrained=False)
        resnet.load_state_dict(mocoModel["state_dict"])

        print("n_inputs :",n_inputs)

        Conv1x1Block = nn.Sequential(
            nn.Conv2d(n_inputs, 3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        # Update input module
        input_module_pre_trained = mocoModel["input_module"]
        conv1x1_default_state_dict = Conv1x1Block.state_dict()
        migrated_data_dict = {}
        for k, v in input_module_pre_trained.items():
            if k == "input_module.net.0.weight":
                if n_inputs == 10:
                    # Set the value only if the n_inputs are 10 (i.e only S2). If they are 12 (both S2 and S1),
                    # the below assignment will result in an error during execution.
                    # Error: "size mismatch for 0.weight: copying a param with shape torch.Size([3, 10, 1, 1]) from checkpoint,
                    # the shape in current model is torch.Size([3, 12, 1, 1])"
                    # The reason is that during pre-training, we have the input set to 10 (for the query block)
                    migrated_data_dict["0.weight"] = input_module_pre_trained["input_module.net.0.weight"]

            elif k == "input_module.net.1.weight":
                migrated_data_dict["1.weight"] = input_module_pre_trained["input_module.net.1.weight"]
            elif k == "input_module.net.1.bias":
                migrated_data_dict["1.bias"] = input_module_pre_trained["input_module.net.1.bias"]
            elif k == "input_module.net.1.running_mean":
                migrated_data_dict["1.running_mean"] = input_module_pre_trained["input_module.net.1.running_mean"]
            elif k == "input_module.net.1.running_var":
                migrated_data_dict["1.running_var"] = input_module_pre_trained["input_module.net.1.running_var"]

        conv1x1_default_state_dict.update(migrated_data_dict)
        Conv1x1Block.load_state_dict(conv1x1_default_state_dict)

        # self.conv1 = nn.Conv2d(n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = nn.Sequential(
            Conv1x1Block,
            # self.conv1,
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )

        self.FC = nn.Linear(2048, numCls)

        # self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        logits = self.FC(x)

        return logits

class Moco_1x1RND(nn.Module):
    def __init__(self, mocoModel, n_inputs = 12, numCls = 17):
        super().__init__()

        resnet = models.resnet50(pretrained=False)

        print("n_inputs :",n_inputs)

        self.Conv1x1Block = nn.Sequential(
            nn.Conv2d(n_inputs, 3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        # self.conv1 = nn.Conv2d(n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = nn.Sequential(
            self.Conv1x1Block,
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )

        self.FC = nn.Linear(2048, numCls)

        # We don't need to initialize here as we are transferring the weights
        #self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        logits = self.FC(x)

        return logits

# This class uses Conv1x1Block block, but it doesn't get initialized from the pre-trained model.
# Only the backbone gets initialized from the pre-trained model
class Moco(nn.Module):
    def __init__(self, mocoModel, n_inputs = 12, numCls = 17):
        super().__init__()

        resnet = models.resnet50(pretrained=False)
        resnet.load_state_dict(mocoModel["state_dict"])

        self.conv1 = nn.Conv2d(n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )

        self.FC = nn.Linear(2048, numCls)

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        logits = self.FC(x)

        return logits

#class ResNet50_em512(nn.Module):
#    def __init__(self, n_inputs = 12, numCls = 17):
#        super().__init__()
#
#        resnet = models.resnet50(pretrained=False)
#
#        self.conv1 = nn.Conv2d(n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#        self.encoder = nn.Sequential(
#            self.conv1,
#            resnet.bn1,
#            resnet.relu,
#            resnet.maxpool,
#            resnet.layer1,
#            resnet.layer2,
#            resnet.layer3,
#            resnet.layer4,
#            resnet.avgpool
#        )
#        self.FC1 = nn.Linear(2048, 512)
#        self.FC2 = nn.Linear(512, numCls)
#
#        self.apply(weights_init_kaiming)
#        self.apply(fc_init_weights)
#
#    def forward(self, x):
#        x = self.encoder(x)
#        x = x.view(x.size(0), -1)
#
#        x = self.FC1(x)
#        logits = self.FC2(x)
#
#        return logits
#
#
#class ResNet50_em(nn.Module):
#    def __init__(self, n_inputs = 12, numCls = 17):
#        super().__init__()
#
#        resnet = models.resnet50(pretrained=False)
#
#        self.conv1 = nn.Conv2d(n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#        self.encoder = nn.Sequential(
#            self.conv1,
#            resnet.bn1,
#            resnet.relu,
#            resnet.maxpool,
#            resnet.layer1,
#            resnet.layer2,
#            resnet.layer3,
#            resnet.layer4,
#            resnet.avgpool
#        )
#        self.FC = nn.Linear(2048, numCls)
#
#        self.apply(weights_init_kaiming)
#        self.apply(fc_init_weights)
#
#    def forward(self, x):
#        x = self.encoder(x)
#        x = x.view(x.size(0), -1)
#
#        logits = self.FC(x)
#
#        return logits, x

class ResNet101(nn.Module):
    def __init__(self, n_inputs = 12, numCls = 17):
        super().__init__()

        resnet = models.resnet101(pretrained=False)

        self.conv1 = nn.Conv2d(n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.FC = nn.Linear(2048, numCls)
        
        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        logits = self.FC(x)

        return logits



class ResNet152(nn.Module):
    def __init__(self, n_inputs = 12, numCls = 17):
        super().__init__()

        resnet = models.resnet152(pretrained=False)

        self.conv1 = nn.Conv2d(n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.FC = nn.Linear(2048, numCls)

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        logits = self.FC(x)

        return logits


if __name__ == "__main__":
    
    inputs = torch.randn((1, 12, 256, 256)) # (how many images, spectral channels, pxl, pxl)

    net = ResNet18()
    #net = ResNet34()
    #net = ResNet50()
    #net = ResNet101()
    #net = ResNet152()

    outputs = net(inputs)

    print(outputs)
    print(outputs.shape)

    numParams = count_parameters(net)

    print(f"{numParams:.2E}")


