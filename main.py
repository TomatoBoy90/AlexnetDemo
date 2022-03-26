import  os
import argparse
from torch.cuda import  amp
from  torch.autograd import  Variable
from  torch.utils.data import  DataLoader
from models.alexnet import  AlexNet
from datasets import  DogCat
from config import *


parser=argparse.ArgumentParser(description="alexnet for pytorch to train class job")
parser.add_argument("--lr",default=0.0001,help="learning rate")
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument("--batch_size",default=128,help="batch size")
parser.add_argument("--amp",action="store_true",help="use amp to train model")
parser.add_argument('--use_dropout', action='store_true', help='use dropout for the alexnet')
parser.add_argument('--mode', default='train', choices=['train','val','test'],help='run mode')
parser.add_argument('--dataroot', type=str, default='./data/dogCat/')
parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')
parser.add_argument('--norm', type=str, default='batch', help='no normalization or batch normalization')
parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
parser.add_argument('--save_latest_freq', type=int, default=10, help='frequency of saving the latest results')
parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
parser.add_argument('--niter', type=int, default=100, help='# iter num')
parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
parser.add_argument("--export_onnx", type=str, help="export ONNX model to a given file")
opt =parser.parse_args()

os.makedirs('pretrained',exist_ok=True)

model=AlexNet().to(device)
train_data=DogCat()
val_data=DogCat(mode='val')
train_dataloader=DataLoader(train_data,batch_size=512,shuffle=True)
val_dataloader=DataLoader(val_data,batch_size=512,shuffle=True)
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
    if(epoch>0&epoch%10==0):
        torch.save(model,'pretrained/'+str(epoch)+'_model.pth')
    val_acc=val(model,val_dataloader)
    print(val_acc)




