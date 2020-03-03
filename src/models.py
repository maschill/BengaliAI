from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.nn.init as init
import torchvision.models as model_zoo
# https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278


class ResClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, width=512):
        super(ResClassifier, self).__init__()
        self.linear1 = nn.Linear(in_channels, width)
        self.linear1_1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, num_classes)
        self.bn1 = nn.BatchNorm1d(width)
        self.bn2 = nn.BatchNorm1d(width)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.linear1(out))
        out = self.bn1(out)
        # out = self.dropout(out)
        # out = F.relu(self.linear1_1(out))
        # out = self.bn2(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out


# 2 3x3 Convolutions with residual connection
class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, width_factor=1, base_width=64):
        super(ResBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2D(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.res_con = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False, stride=stride),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.res_con(x)
        out = self.relu(out)
        
        return out


# 1x1 Conv -> 3x3 Conv -> 1x1 Conv with residual connection
class BottleNeckResBlock(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, width_factor=1, base_width=64):
        super(BottleNeckResBlock, self).__init__()

        width = int(out_channels*base_width/64.)*width_factor #k in wide resnets

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, 3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels*self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.res_con = nn.Sequential(
            nn.Conv2d(in_channels, out_channels*self.expansion, 1, bias=False, stride=stride),
            nn.BatchNorm2d(out_channels*self.expansion)
        )


    def forward(self, x): 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.res_con(x)
        out = self.relu(out)

        return out



class ResNet(nn.Module):
    def __init__(self,in_channels=1, num_classes=(10,10,10)):
        super(ResNet, self).__init__()
        self.width_factor = 1
        self.base_width=64
        base_channels = 32
        self.in_channels = base_channels

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.pooling = nn.MaxPool2d(2, stride=2)

        self.layer1 = self._make_layer(BottleNeckResBlock,base_channels, 3)
        self.layer2 = self._make_layer(BottleNeckResBlock,base_channels*2, 3, stride=2)
        self.layer3 = self._make_layer(BottleNeckResBlock,base_channels*4, 3, stride=2)
        self.layer4 = self._make_layer(BottleNeckResBlock,base_channels*8, 3, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1)) # from Pytorch resnet implementation

        global_pooling_channels = self.in_channels

        self.fc0 = ResClassifier(global_pooling_channels, num_classes[0])
        self.fc1 = ResClassifier(global_pooling_channels, num_classes[1])
        self.fc2 = ResClassifier(global_pooling_channels, num_classes[2])


    def _make_layer(self, block, out_channels, num_blocks, stride=1):  
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride=stride, width_factor=self.width_factor, base_width=self.base_width))
        
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
                layers.append(block(self.in_channels, out_channels, width_factor=self.width_factor, base_width=self.base_width))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pooling(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)

        y0 = self.fc0(out)
        y1 = self.fc1(out)
        y2 = self.fc2(out)

        return y0, y1, y2

class DenseNet(nn.Module): 
    def __init__(self, in_channels, num_classes, net=201):
        self.in_channels = in_channels
        self.num_classes = num_classes
        super(DenseNet, self).__init__()
        self.net = net

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, 3, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(3)
        self.pooling = nn.MaxPool2d(2, stride=2)


        if net == 121:   
            self.densenet = model_zoo.densenet121()
            self.features = self.densenet.features
            global_pooling_channels = 1024    # d201: 1920, d169: 1664,  d121: 1024?


        elif net == 169:  
            self.densenet = model_zoo.densenet169()
            self.features = self.densenet.features
            global_pooling_channels = 1664    # d201: 1920, d169: 1664,  d121: 1024?


        else:  
            self.densenet = model_zoo.densenet201()
            self.features = self.densenet.features
            global_pooling_channels = 1920    # d201: 1920, d169: 1664,  d121: 1024?

        self.fc0 = ResClassifier(global_pooling_channels, num_classes[0], 1024)
        self.fc1 = ResClassifier(global_pooling_channels, num_classes[1], 1024)
        self.fc2 = ResClassifier(global_pooling_channels, num_classes[2], 1024)
    
    def forward(self, x): 
        xx = self.conv1(x)
        xx = self.bn1(xx)
        xx = self.relu(xx)
        xx = self.pooling(xx)

        features = self.features(xx)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1,1))

        y0 = self.fc0(out)
        y1 = self.fc1(out)
        y2 = self.fc2(out)

        return y0, y1, y2



def resnet50(num_classes=(10,10,10)): 
    return ResNet(num_classes=num_classes)

def densenet(num_classes=(10,10,10), in_channels=3, net=201):
    return DenseNet(in_channels=in_channels, num_classes=num_classes, net=net)


def densenet121(num_classes=(10,10,10), in_channels=3):
    return DenseNet(in_channels=in_channels, num_classes=num_classes, net=121)


def densenet169(num_classes=(10,10,10), in_channels=3):
    return DenseNet(in_channels=in_channels, num_classes=num_classes, net=169)