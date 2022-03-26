import torch.nn as nn
import torch

class AlexNetSmallV1(nn.Module):
    def __init__(self,num_classes=1000):
        super(AlexNetSmallV1, self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,
                             out_channels=48,kernel_size=(11,11),
                             stride=(4,4),padding=(2,2)
                             )
        self.norm1 = nn.BatchNorm2d(48)
        self.relu=nn.ReLU(True)
        self.pool1=nn.MaxPool2d(kernel_size=3,
                                stride=2,
                                padding=0,
                                dilation=1)
        self.conv2=nn.Conv2d(in_channels=48,
                             out_channels=128,
                             kernel_size=(3,3),
                             stride=(1,1),padding=(2,2))
        self.norm2 = nn.BatchNorm2d(128)
        self.conv3=nn.Conv2d(in_channels=128,out_channels=192,
                             kernel_size=(3,3),stride=(1,1),
                             padding=(1,1)
                             )
        self.norm3 = nn.BatchNorm2d(192)
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=192,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1)
                               )
        self.conv5 = nn.Conv2d(in_channels=192, out_channels=128,
                               kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1)
                               )
        self.conv6=nn.Conv2d(in_channels=128,out_channels=num_classes,
                             kernel_size=(1,1),stride=(7,7))

    def forward(self,x):
        #1st Layer
        x=self.conv1(x)
        x=self.norm1(x)
        x=self.relu(x)
        #x=self.norm1(x)
        x=self.pool1(x)
        # 2st Layer
        x=self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x=self.pool1(x)
        # 3st Layer
        x=self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        # 4st Layer
        x=self.conv4(x)
        x = self.norm3(x)
        x = self.relu(x)
        # 5st Layer
        x = self.conv5(x)
        x = self.norm2(x)
        x = self.relu(x)
        print(x.shape)
        x=self.conv6(x)[0,:,0,0]
        print(x.shape)
        return x








