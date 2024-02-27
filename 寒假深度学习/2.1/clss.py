import torch
import torch.nn as nn
#ResNet 残差模块，在每层信息传递过程中损失的信息量很小

#残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # 第一个卷积层，用于变换输入的通道数和尺寸
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # 第一个批归一化层
        self.bn1 = nn.BatchNorm2d(out_channels)
        # ReLU 激活函数
        self.relu = nn.ReLU(inplace=True)
        # 第二个卷积层，用于恢复通道数
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # 第二个批归一化层
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 如果步长不为 1 或输入和输出的通道数不同，定义一个 downsample 层来匹配维度
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Identity()  # 如果维度相同且不需要下采样，则使用Identity作为skip connection
    #前向传播过程
    def forward(self, x):
        residual = x
        out = self.conv1(x)     #第一个卷积层的前向传播
        out = self.bn1(out)     #第一个批归一化层的前向传播
        out = self.relu(out)    #ReLU 激活函数
        out = self.conv2(out)   #第二个卷积层的前向传播
        out = self.bn2(out)     #第二个批归一化层的前向传播
        #如果定义了downsample层，进行维度匹配
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual     #将残差加到输出上
        out = self.relu(out)    #LeakyReLU 激活函数
        return out

class ResNet_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet_CNN, self).__init__()

        self.in_channels = 64   #默认初始输入通道数为64
        #第一个卷积层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        #第一个批归一化层
        self.bn1 = nn.BatchNorm2d(64)
        #LeakyReLU 激活函数，有助于解决ReLU的神经元死亡问题
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        #最大池化层
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #构建不同层的残差块
        self.layer1 = self._make_layer(64, 3)    #构建了一个包含3个残差块的残差块组，每个残差块的输入通道数为64，输出通道数也为64
        self.layer2 = self._make_layer(128, 4, stride=2)    #构建了一个包含4个残差块的残差块组，每个残差块的输入通道数为64（由上一层的输出通道数决定），输出通道数为128，且采用步长为2的卷积操作
        self.layer3 = self._make_layer(256, 6, stride=2)    #构建了一个包含6个残差块的残差块组，每个残差块的输入通道数为128（由上一层的输出通道数决定），输出通道数为256，且采用步长为2的卷积操作
        self.layer4 = self._make_layer(512, 3, stride=2)    #构建了一个包含3个残差块的残差块组，每个残差块的输入通道数为256（由上一层的输出通道数决定），输出通道数为512，且采用步长为2的卷积操作
        #全局平均池化层
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #Dropout 层，处理过拟合
        self.dropout = nn.Dropout(0.3)
        #全连接层
        self.fc = nn.Linear(512, num_classes)

    #根据输入的输出通道数和残差块数量构建一个由多个残差块构成的层
    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels)) # 逐层增加通道数
            layers.append(nn.BatchNorm2d(out_channels))  # 在每个残差块后添加BatchNorm2d层
        return nn.Sequential(*layers)

    #前向传播过程
    def forward(self, x):
        x = self.conv1(x)  #第一个卷积层的前向传播
        x = self.bn1(x)    #第一个批归一化层的前向传播
        x = self.relu(x)   #LeakyReLU 激活函数
        x = self.maxpool(x)  #最大池化层
        x = self.layer1(x)  #第一个残差块组
        x = self.layer2(x)  #第二个残差块组
        x = self.layer3(x)  #第三个残差块组
        x = self.layer4(x)  #第四个残差块组
        x = self.global_avgpool(x)  #全局平均池化
        x = x.view(x.size(0), -1)   #展平特征向量
        x = self.dropout(x)          #Dropout
        x = self.fc(x)               #全连接层


        return x

