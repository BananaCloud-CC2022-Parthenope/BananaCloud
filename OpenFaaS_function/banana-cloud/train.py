import torch
import time
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt

from torchsummary import summary


#from models.mlp import MLP
from statenet import StateNet
from alexnet import AlexNet
from resnet import resnet18, resnet34, resnet50
from vgg import vgg16
from utility.pad import NewPad, Equalize

import argparse
from tensorboardX import SummaryWriter
import math
import sys
import os

parser = argparse.ArgumentParser()
#parser.add_argument('--network', type = str, choices = ['resnet', 'sqnxt'], default = 'sqnxt') #da definire
parser.add_argument('--num_epochs', type = int, default = 150)
parser.add_argument('--lr', type=float, default = 0.1)
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--nm_conv', type=str)
args = parser.parse_args()

# Inserire modello da importare in base ad args.network

writer = SummaryWriter('board/statenet_cnn_' + args.nm_conv)

num_epochs = int(args.num_epochs)
lr           = float(args.lr)
start_epoch  = 0
batch_size   = int(args.batch_size)

is_use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_use_cuda else "cpu")
best_acc    = 0.

#Dataset Preprocessing
data_dir = 'tools/dataset/Banana_Dataset/cropped/'

# conf = 1
# for filename in os.listdir('checkpoints/resume/'):
#     if filename == f'checkpoint_{args.nm_conv}conv_conf_{str(conf)}.pth':
#         conf += 1

train_transform = transforms.Compose([
    NewPad(),
    transforms.Resize((256,256)),
    transforms.RandomCrop((227,227)),
    #Equalize(),
    transforms.ToTensor(),
    transforms.Normalize([0.313, 0.300, 0.253],[ 0.343, 0.336, 0.315 ]) #RGB order
])

dataset = ImageFolder(data_dir,transform = train_transform)
img, label = dataset[0]
print(img.shape,label)

plt.hist(img.numpy().ravel(), bins=30, density=True)
plt.xlabel("pixel values")
plt.ylabel("relative frequency")
plt.title("distribution of pixels")
plt.savefig('hist_img.png')


print("Following classes are there : \n", dataset.classes)

train_size = int(len(dataset) * 0.8)
val_size = int(len(dataset) * 0.1)
test_size = len(dataset) - train_size - val_size

train_data, val_data, test_data = random_split(dataset,[train_size,val_size,test_size])

print(f"Length of Train Data : {len(train_data)}")
print(f"Length of Validation Data : {len(val_data)}")
print(f"Length of Test Data : {len(test_data)}")

train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
val_dl = DataLoader(val_data, batch_size*2, num_workers = 4, pin_memory = True)
test_dl = DataLoader(test_data, batch_size, num_workers = 4, pin_memory = True)

def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, 0.0, 0.01)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.01)
 


#model = MLP(529200, 100000)
#model = resnet34()
model = vgg16()
#model = AlexNet()
#model.apply(weights_init)
#model = StateNet()
#model = torch.hub.load('pytorch/vision:v.0.10.0', 'densenet121', pretrained=False)

if is_use_cuda:
    model.to(device)
    model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

print('stampo summary')
summary(model, input_size=(3, 227, 227))

#da valutarle
criterion = nn.CrossEntropyLoss()
optimizer  = optim.SGD(model.parameters(), lr = lr, momentum = 0.9, weight_decay = 5e-4)

loss_a = []



def train(epoch):
    model.train()

    train_loss = 0
    correct = 0
    total = 0
    it = 0

    print('Training epoch: #%d'%(epoch+1))
    
    for idx, (input,labels) in enumerate(train_dl):
        if is_use_cuda:
            inp, target = input.to(device), labels.to(device) 
        
        optimizer.zero_grad()
        output = model(inp)
        #output = torch.nn.functional.softmax(output, dim=1)
        loss = criterion(output, target)
        loss.backward()
        

        optimizer.step()

        train_loss += loss.item()
        _, predict = torch.max(output, 1)

        total += target.size(0)
        correct += predict.eq(target).cpu().sum().double()

        sys.stdout.write('\n')
        sys.stdout.write('[%s] Training Epoch [%d/%d] Iter[%d/%d] \t\t Loss: %.4f Acc@1: %.3f'
                        %(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                        epoch+1, num_epochs, idx, len(train_dl), loss.item(), correct / total))
        sys.stdout.flush()

    checkpoint = {
    'state_dict': model.module.state_dict(),
    'optimizer': optimizer.state_dict()
    }

    torch.save(checkpoint, f'checkpoints/resume/checkpoint_{args.nm_conv}.pth')
    
    sys.stdout.write('\n')
    sys.stdout.write('\n\n[%s] Training Epoch [%d/%d]\t\tAvg Loss: %.4f Avg Acc: %.3f'
                    % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                       epoch+1, num_epochs, train_loss/len(train_dl), correct/total))
    sys.stdout.flush()
    writer.add_scalar('Train/Loss', train_loss/len(train_dl), epoch)
    writer.add_scalar('Train/Accuracy', correct / total, epoch )

best_acc = 0
best_loss = 999

def test(epoch, b_e, dl, text):
    global best_acc
    global best_loss
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    avg_loss = 0
    avg_acc = 0

    for idx, (input,labels) in enumerate(dl):
        if is_use_cuda:
            inp, target = input.to(device), labels.to(device) 
        
        output = model(inp)
        #output = torch.nn.functional.softmax(output, dim=1)
        loss = criterion(output, target)

        test_loss  += loss.item()
        _, predict = torch.max(output, 1)
        total += target.size(0)
        correct += predict.eq(target).cpu().sum().double()

        sys.stdout.write('\n')
        if text == 'Validation':
            sys.stdout.write('[%s] %s Epoch [%d/%d] Iter[%d/%d]\t\tLoss: %.4f Acc@1: %.3f'
                        % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), text,
                           epoch+1, num_epochs, idx, test_size // dl.batch_size,
                          loss.item(), correct / total))
        else:
            sys.stdout.write('[%s] %s\t Epoch %d \tLoss: %.4f Acc@1: %.3f'
                        % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), text, b_e,
                          loss.item(), correct / total))
        sys.stdout.flush()
    if text == 'Validation':
        avg_loss = test_loss / len(dl)
        avg_acc = correct / total
        if avg_acc > best_acc and avg_loss <= best_loss:
            best_acc = avg_acc
            best_loss = avg_loss
            b_e = epoch
            checkpoint = {
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            print("\nBest model saved. Best acc = %.3f Best Loss: %.4f" % (avg_acc, avg_loss))
            torch.save(checkpoint, f'checkpoints/best/best_checkpoint_{args.nm_conv}.pth')
            test(0, b_e, test_dl, 'Testing')

    writer.add_scalar(text+'/Loss', test_loss/len(dl), epoch)
    writer.add_scalar(text+'/Accuracy', correct / total, epoch)
    return b_e


b_e = 0
for _epoch in range(start_epoch, start_epoch + num_epochs):
    start_time = time.time()
    train(_epoch)
    print()
    b_e = test(_epoch, b_e, val_dl ,'Validation')
    print()
    print()
    end_time = time.time()
    print('Epoch #%d Cost %ds' % (_epoch, end_time - start_time))

writer.close()









