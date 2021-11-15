import os 
import time 
import json 
import argparse
import torch 
import torchvision
import random
import numpy as np 
from data import FaceDataset
from tqdm import tqdm 
from torch import nn
from torch import optim
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from model.resnet50_ft_dims_2048 import resnet50_ft
import cv2
import torch.nn.functional as F
import csv
LAMBDA_1 = 0.2
LAMBDA_2 = 0.05
START_AGE = 0
END_AGE = 100
VALIDATION_RATE= 0.1

def vgg_block(in_channels, out_channels, more=False):
    blocklist = [
        ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)),
        ('relu1', nn.ReLU(inplace=True)),
        ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)),
        ('relu2', nn.ReLU(inplace=True)),
    ]
    if more:
        blocklist.extend([
            ('conv3', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)),
            ('relu3', nn.ReLU(inplace=True)),
        ])
    blocklist.append(('maxpool', nn.MaxPool2d(kernel_size=2, stride=2)))
    block = nn.Sequential(OrderedDict(blocklist))
    return block
class VGG(nn.Module):
    def __init__(self, classes=1000, channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            vgg_block(channels, 64),
            vgg_block(64, 128),
            vgg_block(128, 256, True),
            vgg_block(256, 512, True),
            vgg_block(512, 512, True),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5, inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5, inplace=True),
        )
        self.cls = nn.Linear(4096, 101)

    def forward(self, x):
        in_size = x.shape[0]
        x = self.conv(x)
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.cls(x)
        x = F.softmax(x, dim=1)
        return x
class VGG16_AGE(VGG):
    def __init__(self, classes=101, channels=3):
        super().__init__()
        self.cls = nn.Linear(4096, 101)

def predict(model, image):

    model.eval()
    with torch.no_grad():
        #image = image.astype(np.float32)
      
        image = np.transpose(image, (2,0,1))
        img = torch.from_numpy(image).cuda()
        img = img.type('torch.FloatTensor').cuda()
       
        output = model(img[None])
        # m = nn.Softmax(dim=1)
        
        # output = m(output)
        
        a = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).cuda()
        mean = (output * a).sum(1, keepdim=True).cpu().data.numpy()
        pred = np.around(mean)[0][0]
    return pred


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_directory', type=str)
    parser.add_argument('-pm', '--pred_model', type=str, default='./weights/mean_variance_ffhq/model_best_loss')
    parser.add_argument('-path','--pred_path',type=str,default='../result/10')
    parser.add_argument('-out','--outcsv',type=str,default='./test_result/test')
    return parser.parse_args()


def main():
    
    args = get_args()
    
    if args.pred_path and args.pred_model:
        all_num = 0
        all_MAE = 0
        all_age = 0
        model = VGG16_AGE()
        # model = resnet50_ft()
        model.load_state_dict(torch.load(args.pred_model))
        model.eval()
        model.cuda()
        with open(args.outcsv+'.csv', 'a+', newline='') as csvfile:
            for filename in os.listdir(args.pred_path):
                img = cv2.imread(args.pred_path+"/"+filename)
                resized_img = cv2.resize(img, (224, 224))
                pred = predict(model, resized_img) 
                writer =  csv.writer(csvfile)
                writer.writerow([filename,pred])
                all_age+= pred
                all_num+=1

if __name__ == "__main__":
    main()
