import torch
import torch.nn as nn
import torchvision
from torchvision import datasets ,transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# device config
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size =784 # 28x28
hidden_size = 100
num_classes =10
num_epochs =4
batch_size =100
learning_rate =0.001

mean = np.array([0.485,0.456,0.406])
std = np.array([0.229,0.224,0.225])

data_transforms={
    'train':transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

data_dir = './FaceR_Dataset'
sets = ['train','val']
img_dataset = {x:datasets.ImageFolder(os.path.join(data_dir,x),
                                        data_transforms[x])
                 for x in sets}
img_loader = {x:torch.utils.data.DataLoader(img_dataset[x],batch_size=4,
                                            shuffle=True, num_workers=0)
              for x in sets}
examples = iter(img_loader)
print(examples)
sample ,labels =next(examples)
print(sample.shape , labels.shape)
print(type(sample) ,type(labels))

# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.imshow(sample[i][0] , cmap='gray')
# plt.show()
