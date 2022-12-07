# We reference the resnet implementation from here "https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py".

'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, image_size=32):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_proj = ProjHead((16 * image_size * image_size), 256, 256)
        
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.block1_proj = ProjHead((16 * image_size * image_size), 256, 256)
        
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        image_size = image_size // 2
        self.block2_proj = ProjHead((32 * image_size * image_size), 256, 256)
        
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        image_size = image_size // 2
        self.block3_proj = ProjHead((64 * image_size * image_size), 256, 256)
        
        self.linear = nn.Linear(64, num_classes)
        
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.relu(self.bn1(self.conv1(x)))
        conv1_out = self.conv1_proj(torch.flatten(out, 1))
        
        out = self.layer1(out)
        block1_out = self.block1_proj(torch.flatten(out, 1))

        out = self.layer2(out)
        block2_out = self.block2_proj(torch.flatten(out, 1))

        out = self.layer3(out)
        block3_out = self.block3_proj(torch.flatten(out, 1))

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        intermediate_out = {
            'conv1': conv1_out,
            'block1': block1_out,
            'block2': block2_out,
            'block3': block3_out,
        }
        
        return out, intermediate_out

# out = F.relu(self.bn1(self.conv1(x)))
# out = F.avg_pool2d(out, out.size()[3])
# out = out.view(out.size(0), -1)

def resnet20(**kwargs):
    return ResNet(BasicBlock, [3, 3, 3], **kwargs)


def resnet32(**kwargs):
    return ResNet(BasicBlock, [5, 5, 5], **kwargs)


def resnet44(**kwargs):
    return ResNet(BasicBlock, [7, 7, 7], **kwargs)


def resnet56(**kwargs):
    return ResNet(BasicBlock, [9, 9, 9], **kwargs)


def resnet110(**kwargs):
    return ResNet(BasicBlock, [18, 18, 18], **kwargs)


def resnet1202(**kwargs):
    return ResNet(BasicBlock, [200, 200, 200], **kwargs)


class ProjHead(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(ProjHead, self).__init__()
        
        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, out_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        
        return out
        

# class CNNSmall(nn.Module):
#     def __init__(self, num_classes=10):
#         super(CNN, self).__init__()
        
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.conv1_proj = ProjHead((6 * 14 * 14), 256, 256)
        
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.conv2_proj = ProjHead((16 * 5 * 5), 256, 256)
        
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(2, 2)

#         self.fc1 = nn.Linear((16 * 5 * 5), 120)
#         self.fc2 = nn.Linear(120, 84)        
#         self.fc_proj = ProjHead(84, 84, 256)
        
#         self.fc3 = nn.Linear(84, num_classes)

#     def forward(self, x):

#         out = self.maxpool(self.relu(self.conv1(x)))
#         conv1_out = self.conv1_proj(torch.flatten(out, 1))
        
#         out = self.maxpool(self.relu(self.conv2(out)))
#         conv2_out = self.conv2_proj(torch.flatten(out, 1))
        
#         out = out.view(-1, 16 * 5 * 5)

#         out = self.relu(self.fc1(out))
#         out = self.relu(self.fc2(out))
#         fc_out = self.fc_proj(out)
        
#         out = self.fc3(out)
        
#         intermediate_out = {
#             'conv1': conv1_out,
#             'conv2': conv2_out,
#             'fc': fc_out,
#         }
        
#         return out, intermediate_out


# def cnn_small(**kwargs):
#     return CNNSmall(**kwargs)


# class CNNLarge(nn.Module):
#     def __init__(self, num_classes=10):
#         super(CNN, self).__init__()
        
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=7)
#         self.conv1_proj = ProjHead((16 * 26 * 26), 256, 256)

#         self.conv2 = nn.Conv2d(16, 32, kernel_size=7)
#         self.conv2_proj = ProjHead((32 * 20 * 20), 256, 256)
        
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=7)
#         self.conv3_proj = ProjHead((64 * 14 * 14), 256, 256)
        
#         self.conv4 = nn.Conv2d(64, 64, kernel_size=7)
#         self.conv4_proj = ProjHead((64 * 8 * 8), 256, 256)
        
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(2, 2)
        
#         self.fc1 = nn.Linear((64 * 8 * 8), 512)
#         self.fc1_proj = ProjHead(512, 256, 256)
        
#         self.fc2 = nn.Linear(512, 256)        
#         self.fc2_proj = ProjHead(256, 128, 256)
        
#         self.fc3 = nn.Linear(256, num_classes)

#     def forward(self, x):

#         out = self.relu(self.conv1(x))
#         conv1_out = self.conv1_proj(torch.flatten(out, 1))
        
#         out = self.relu(self.conv2(out))
#         conv2_out = self.conv2_proj(torch.flatten(out, 1))
        
#         out = self.relu(self.conv3(out))
#         conv3_out = self.conv3_proj(torch.flatten(out, 1))
        
#         out = self.relu(self.conv4(out))
#         conv4_out = self.conv4_proj(torch.flatten(out, 1))
        
#         out = out.view(-1, 64 * 8 * 8)

#         out = self.relu(self.fc1(out))
#         fc1_out = self.fc1_proj(out)
        
#         out = self.relu(self.fc2(out))
#         fc2_out = self.fc2_proj(out)
        
#         out = self.fc3(out)

#         intermediate_out = {
#             'conv1': conv1_out,
#             'conv2': conv2_out,
#             'conv3': conv3_out,
#             'conv4': conv4_out,
#             'fc1': fc1_out,
#             'fc2': fc2_out,
#         }
        
#         return out, intermediate_out
    

# def cnn_large(**kwargs):
#     return CNNLarge(**kwargs)


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3)
        self.conv1_proj = ProjHead((8 * 15 * 15), 256, 256)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv2_proj = ProjHead((16 * 6 * 6), 256, 256)
        
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3_proj = ProjHead((32 * 2 * 2), 256, 256)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear((32 * 2 * 2), 128)
        self.fc1_proj = ProjHead(128, 256, 256)
        
        self.fc2 = nn.Linear(128, 96)        
        self.fc2_proj = ProjHead(96, 96, 256)
        
        self.fc3 = nn.Linear(96, num_classes)

    def forward(self, x):

        out = self.maxpool(self.relu(self.conv1(x)))
        conv1_out = self.conv1_proj(torch.flatten(out, 1))
        
        out = self.maxpool(self.relu(self.conv2(out)))
        conv2_out = self.conv2_proj(torch.flatten(out, 1))
        
        out = self.maxpool(self.relu(self.conv3(out)))
        conv3_out = self.conv3_proj(torch.flatten(out, 1))
        
        out = out.view(-1, 32 * 2 * 2)

        out = self.relu(self.fc1(out))
        fc1_out = self.fc1_proj(out)
        
        out = self.relu(self.fc2(out))
        fc2_out = self.fc2_proj(out)
        
        out = self.fc3(out)

        intermediate_out = {
            'conv1': conv1_out,
            'conv2': conv2_out,
            'conv3': conv3_out,
            'fc1': fc1_out,
            'fc2': fc2_out,
        }
        
        return out, intermediate_out

    
def cnn(**kwargs):
    return CNN(**kwargs)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
