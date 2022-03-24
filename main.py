
import torch
from  torch.autograd import  Variable
from  torch.utils.data import  DataLoader
from  alexnet import  AlexNet
from datasets import  DogCat

device='cuda'if torch.cuda.is_available() else'cpu'
per=1
model=AlexNet().to(device)
train_data=DogCat()
val_data=DogCat(mode='val')
train_dataloader=DataLoader(train_data,batch_size=1,shuffle=True)
val_dataloader=DataLoader(val_data,batch_size=1,shuffle=True)
cel=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.0001)

def val(model,dataloader):
    model.eval()
    num_total=0
    err=0
    for img,label in dataloader:
        with torch.no_grad():
            img=Variable(img).to(device)
            label = Variable(label).to(device)
            _,pred=torch.max(model(img).data,1)
            num_total+=label.size(0)
            err+=(pred == label).sum().item()
            acc=100*(err/num_total)
            model.train()
            return acc

for epoch in range(100):
    for i,(image,label) in enumerate(train_dataloader):
        image,label=Variable(image.to(device)),Variable(label.to(device))
        loss=cel(model(image),label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if(epoch>0&epoch%per==0):
        #torch.save(model,str(epoch)+'_model.pth')
        val_acc=val(model,val_dataloader)
        print(val_acc)




