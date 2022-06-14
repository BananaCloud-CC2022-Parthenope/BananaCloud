import torch
import time
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import cv2 as cv
from torchvision import transforms
import torch.nn.functional as F
import sys
import os

from pathlib import Path


#os.chdir('../')
print('test.py: ',Path.cwd())
from .utility.pad import NewPad
#sys.path.append('/home/quaternione/FruitsEvaluationNet-main/models')
#from .models.resnet import resnet18, resnet34, resnet50
from .models.alexnet import AlexNet
from PIL import Image

import math

import os


class FruitsEvaluationNet():
    def __init__(self):
        self.net = AlexNet()
        self.test_transform = transforms.Compose([
            NewPad(),
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.313, 0.300, 0.253],
            std=[0.343, 0.336, 0.315]
            )
        ])
        self.classes = ('underipe', 'barely ripe', 'ripe', 'very ripe', 'overipe', 'rotten')

        self.load_checkpoints()
        print(self.classes)
        
        
    def load_checkpoints(self):
        root = '/home/app/function/core/'
        path = os.path.join(root, 'best_checkpoint_alexnet_256b_256.pth')

        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.net.load_state_dict(checkpoint['state_dict'])
        print("\nLoading checkpoint complete")


    def test(self, img):
        self.net.eval()
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.test_transform(img)
        img = img.unsqueeze(0)
        output = self.net(img)

        output = F.softmax(output, dim=1)

        _, pred_label_idx = torch.max(output, 1)
        print('classe predetta: ', self.classes[pred_label_idx])

        return self.classes[pred_label_idx]