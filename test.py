import os
import cv2
import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import  DataLoader
from  alexnet import AlexNet
from datasets import  DogCat

os.makedirs('results/dogCat/',exist_ok=True)
test_data=DogCat(mode='test')
test_dataloader=DataLoader(test_data,batch_size=1,shuffle=True)
model=AlexNet()
model.load_state_dict(torch.load('dogCat_30.pth',map_location=torch.device('cpu')))
model.eval()

for i,img in enumerate(test_dataloader):
    img=Variable(img)
    with torch.no_grad():
        model(img)
        _, pred = torch.max(model(img).data, 1)
        label='dog' if pred==1 else 'cat'
        torchvision.utils.save_image(img,'results/dogCat/'+str(i)+'---'+label+'.jpg')


