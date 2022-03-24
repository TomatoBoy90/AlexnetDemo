
import os
from  PIL import  Image
import  torch
from torch.utils import  data
from  torchvision import  transforms

device=' cuda'if torch.cuda.is_available() else'cpu'
class DogCat(data.Dataset):
    def __init__(self,root='./data/dogCat/',mode='train'):
        super(DogCat, self).__init__()
        self.mode=mode
        self.train_path=root+'train/'
        self.test_path = root + 'test/'
        self.val_path = root + 'val/'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if mode=='train':
            self.train_images=[os.path.join(root+'/train/',i) for i in os.listdir(self.train_path)][:]
            self.train_images.sort()
            self.transforms = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])
        elif mode=='test':
            self.test_images = [os.path.join(root+ 'test/', i) for i in os.listdir(self.test_path)]
            self.test_images.sort()
            self.transforms = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                normalize])
        elif mode=='val':
            self.val_images = [os.path.join(root+ 'val/', i) for i in os.listdir(self.val_path)]
            self.val_images.sort()
            self.transforms = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                normalize])
    def __getitem__(self, index):
        if self.mode=='train':
             _path=self.train_images[index]
             image=Image.open(_path)
             label=1 if 'dog' in _path.split('/')[-1] else 0
             image=self.transforms(image)
             return image,label
        elif self.mode=='val':
            _path = self.val_images[index]
            image = Image.open(_path)
            label = 1 if 'dog' in _path.split('/')[-1] else 0
            image = self.transforms(image)
            return image, label

    def __len__(self):
        if self.mode=='train':
            return len(self.train_images)
        elif self.mode == 'val':
            return len(self.val_images)

