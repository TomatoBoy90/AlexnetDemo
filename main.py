
import argparse
import torch
from torch.cuda import  amp
from  torch.autograd import  Variable
from  torch.utils.data import  DataLoader
from  alexnet import  AlexNet
from datasets import  DogCat
from config import *


parser=argparse.ArgumentParser(description="alexnet for pytorch to train class job")
parser.add_argument("-lr",default=0.0001,help="learning rate")
parser.add_argument("-batch_size",default=1,help="batch size")
parser.add_argument("-amp",action="store_true",help="use amp to train model")
args =parser.parse_args()



model=AlexNet().to(device)
train_data=DogCat()
val_data=DogCat(mode='val')
train_dataloader=DataLoader(train_data,batch_size=1,shuffle=True)
val_dataloader=DataLoader(val_data,batch_size=1,shuffle=True)
cel=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.0001)

def val(model,dataloader):
    model.eval()
    num_total=0.0
    err=0.0
    for img,label in dataloader:
        with torch.no_grad():
            img=Variable(img).to(device)
            label = Variable(label).to(device)
            _,pred=torch.max(model(img).data,1)
            num_total+=label.size(0)
            err+=(pred == label).sum().item()
            acc=100*(err/num_total*1.0)
            model.train()
            return acc
scalar=amp.GradScaler(True)
for epoch in range(epochs):
    for i,(image,label) in enumerate(train_dataloader):
        image,label=Variable(image.to(device)),Variable(label.to(device))
        with amp.autocast(True):
            loss=cel(model(image),label)
        scalar.scale(loss).backward()
        scalar.step(optimizer)
        scalar.update()
        optimizer.zero_grad()
    if(epoch>0&epoch%per==0):
        #torch.save(model,str(epoch)+'_model.pth')
        val_acc=val(model,val_dataloader)
        print(val_acc)




