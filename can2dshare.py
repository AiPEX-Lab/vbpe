import numpy as np 
import torch 
import torch.nn as nn
import cv2
import os 
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import random

def resize_video(videopath):
    cap = cv2.VideoCapture(videopath)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    X_data = []
    for j in range(length): 
        ret, frame = cap.read()
        frame = cv2.resize(frame,(36,36),interpolation=cv2.INTER_AREA)
        frames.append(frame)
    X_data.append(frames)
    X_data = torch.squeeze(torch.from_numpy(np.asarray(X_data)))
    X_data = X_data.permute(0, 3, 1, 2)
    return X_data

def get_appearance_motion(image_batch):
    assert image_batch.shape[1:] == torch.Size([3, 36, 36]) and len(image_batch.shape) == 4
    lshifted = torch.cat([image_batch, torch.zeros(1,3,36,36)], 0)[1:]
    motion = (image_batch//2 - lshifted//2) / (image_batch//2 + lshifted//2 + 1e-5)
    motion[motion > 3] = 3
    motion[motion < -3] = -3
    motion = motion[:-1]
    motion = (motion - motion.mean()) / motion.std()
    appearance = image_batch[:-1]
    estimated_mean, estimated_std = torch.mean(torch.Tensor.float(appearance)), torch.std(torch.Tensor.float(appearance))
    appearance = (appearance - estimated_mean) / estimated_std
    return appearance, motion

class CAN_2d(nn.Module):
    def __init__(self, pretrained=True):
        super(CAN_2d, self).__init__()

        self.conv1_motion = nn.Conv2d(3, 32, (3, 3), padding=1)
        self.conv2_motion = nn.Conv2d(32, 32, (3, 3))
        self.avgpool1_motion = nn.AvgPool2d((2, 2))
        self.dropout1_motion = nn.Dropout(0.25)
        self.conv3_motion = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.conv4_motion = nn.Conv2d(64, 64, (3, 3))
        self.avgpool2_motion = nn.AvgPool2d((2, 2))
        self.dropout2_motion = nn.Dropout(0.25)
        self.dense1_motion = nn.Linear(3136, 128)
        self.dropout3_motion = nn.Dropout(0.5)
        self.dense2_motion = nn.Linear(128, 1)

        # ~~

        self.conv1_appearance = nn.Conv2d(3, 32, (3, 3), padding=1)
        self.conv2_appearance = nn.Conv2d(32, 32, (3, 3))
        self.conv2_attention = nn.Conv2d(32, 1, (1, 1))   # ***

        self.avgpool1_appearance = nn.AvgPool2d((2, 2))
        self.dropout1_appearance = nn.Dropout(0.25)
        self.conv3_appearance = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.conv4_appearance = nn.Conv2d(64, 64, (3, 3))
        self.conv4_attention = nn.Conv2d(64, 1, (1, 1)) # ***
        
        
        # ~~

        if pretrained:
            self.load()
            
        self.pretrained = pretrained


    def masknorm(self, x):
        xsum = torch.sum(torch.sum(x, axis=2, keepdims=True), axis=3, keepdims=True)
        xshape = x.shape
        ans = (x/xsum)*xshape[2]*xshape[3]*0.5
        return ans

    def forward(self, xm, xa):        

        debug = {}
        
        xm = torch.tanh(self.conv1_motion(xm))
        
        xm = torch.tanh(self.conv2_motion(xm))
        

        # ***
        xa = torch.tanh(self.conv1_appearance(xa))
        xa = torch.tanh(self.conv2_appearance(xa))
        ga = self.masknorm(torch.sigmoid(self.conv2_attention(xa)))
        debug['mask1'] = ga

        # ***

        xm = xm * ga
        xm = self.avgpool1_motion(xm)
        
        xm = self.dropout1_motion(xm)
        

        xm = torch.tanh(self.conv3_motion(xm))
        
        xm = torch.tanh(self.conv4_motion(xm))
        

        # ***
        xa = self.avgpool1_appearance(xa)
        xa = self.dropout1_appearance(xa)
        xa = torch.tanh(self.conv3_appearance(xa))
        xa = torch.tanh(self.conv4_appearance(xa))
        ga = self.masknorm(torch.sigmoid(self.conv4_attention(xa)))
        debug['mask2'] = ga
        # ***

        xm = xm * ga
        xm = self.avgpool2_motion(xm)
        
        xm = self.dropout2_motion(xm)
        

        xm = xm.permute(0, 2, 3, 1)
        xm = torch.flatten(xm, 1)
        xm = torch.tanh(self.dense1_motion(xm))
        

        xm = self.dropout3_motion(xm)
        debug['dense1'] = xm
        xm = self.dense2_motion(xm)
        

        output = xm
        #print(output)
        return output, xa, debug


    def load_weights_from_keras(self, kmodel):
        ws = kmodel.get_weights()
        load_layer(self.conv1_appearance, ws[0], ws[1])
        load_layer(self.conv2_appearance, ws[2], ws[3])
        load_layer(self.conv1_motion, ws[4], ws[5])
        load_layer(self.conv2_attention, ws[6], ws[7]) # ***
        load_layer(self.conv2_motion, ws[8], ws[9]) 

        load_layer(self.conv3_appearance, ws[10], ws[11])
        load_layer(self.conv4_appearance, ws[12], ws[13])
        load_layer(self.conv3_motion, ws[14], ws[15])
        load_layer(self.conv4_attention, ws[16], ws[17]) # ***
        load_layer(self.conv4_motion, ws[18], ws[19])

        load_layer(self.dense1_motion, ws[20], ws[21], (1, 0))
        load_layer(self.dense2_motion, ws[22], ws[23], (1, 0))
    
    def load(self, path=r'./can2d_pytorch.pth'):
         self.load_state_dict(torch.load(path))


def deep_phys(mp4_path, output_ppg_path):
    can = CAN_2d()
    path = mp4_path
    for root,dirs,files in os.walk(path):
        videonames=[ _ for _ in files if _.endswith('.mp4') ]
    outputhr = []
    bvp = []
    for i in range(0, len(videonames)):
        vidpath = os.path.join(path, videonames[i])
        print(vidpath)
        video_proc = resize_video(vidpath)
        output = can(get_appearance_motion(video_proc)[1], get_appearance_motion(video_proc)[0])
        output_ppg_name = output_ppg_path + 'Result_' + os.path.splitext(videonames[i])[0] + '.csv'
        
        print(output_ppg_name)
        np.savetxt(output_ppg_name, output[0].detach().numpy())
        