import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self,num_classes=1000):
        super(AlexNet, self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,
                             out_channels=48,kernel_size=(11,11),
                             stride=(4,4),padding=(2,2)
                             )
        self.relu=nn.ReLU(True)
        self.norm1=nn.LocalResponseNorm(size=2)
        self.pool1=nn.MaxPool2d(kernel_size=3,
                                stride=2,
                                padding=0,
                                dilation=1)

        self.conv2=nn.Conv2d(in_channels=48,
                             out_channels=128,
                             kernel_size=(5,5),
                             stride=(1,1),padding=(2,2))
        self.conv3=nn.Conv2d(in_channels=128,out_channels=192,
                             kernel_size=(3,3),stride=(1,1),
                             padding=(1,1)
                             )
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=192,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1)
                               )
        self.conv5 = nn.Conv2d(in_channels=192, out_channels=128,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1)
                               )

        self.fc1 = nn.Linear(13*13*128,2048,bias=True)
        self.fc2 = nn.Linear(2048, 2048, bias=True)
        self.fc3 = nn.Linear(2048, num_classes, bias=True)
        self.drop=nn.Dropout(0.5)

    def forward(self,x):
        #1st Layer
        x=self.conv1(x)
        x=self.relu(x)
        x=self.norm1(x)
        x=self.pool1(x)
        # 2st Layer
        x=self.conv2(x)
        x = self.relu(x)
        x=self.norm1(x)
        x=self.pool1(x)
        # 3st Layer
        x=self.conv3(x)
        x = self.relu(x)
        # 4st Layer
        x=self.conv4(x)
        x = self.relu(x)
        # 5st Layer
        x = self.conv5(x)
        x = self.relu(x)
        x = x.view(x.size(0), 128 * 13 * 13)
        x=self.drop(x)
        x=self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x=self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x










