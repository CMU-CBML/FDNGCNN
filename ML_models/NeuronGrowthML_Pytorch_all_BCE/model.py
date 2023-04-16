import torch
import torch.nn as nn
import torch.nn.init as init

import matplotlib.pyplot as plt
import numpy as np

class rdcnn_2_larger(nn.Module):
    def __init__(self, drop_rate):
        super(rdcnn_2_larger, self).__init__()
        self.encoder  = nn.Sequential(
            nn.Conv2d(3, 84, 4, stride=2, padding=1),  # b, 84, 11, 11
            nn.BatchNorm2d(84),
            nn.ReLU(True),
            nn.Dropout(drop_rate) ,

            nn.Conv2d(84, 168, 4, stride=2, padding=1),  # b, 168, 6, 6
            nn.BatchNorm2d(168),
            nn.ReLU(True),
            nn.Dropout(drop_rate) ,

            nn.MaxPool2d(2, stride=1),  # b, 168, 5, 5
            nn.Conv2d(168, 336, 4, stride=2, padding=1),  # b, 336, 3, 3
            nn.BatchNorm2d(336),
            nn.ReLU(True),
            nn.Dropout(drop_rate) ,

            nn.MaxPool2d(2, stride=1),  # b, 336, 2, 2
            nn.Dropout(drop_rate) ,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(336, 672, 6, stride=2, padding=1),  # b, 672, 3, 3
            nn.BatchNorm2d(672),
            nn.ReLU(True),
            nn.ConvTranspose2d(672, 336, 5, stride=2),  # b, 336, 6, 6
            nn.BatchNorm2d(336),
            nn.ReLU(True),
            nn.ConvTranspose2d(336, 84, 5, stride=2),  # b, 84, 12, 12
            nn.BatchNorm2d(84),
            nn.ReLU(True),
            nn.ConvTranspose2d(84, 1, 4, stride=1,padding=2),  # b, 1, 21, 21
        )

    def forward(self, x):
        x = self.encoder(x)
        # tmp = x.detach().cpu().numpy()
        # print(tmp.shape)
        # plt.imshow(tmp[10,10,:,:],cmap='jet')
        # plt.colorbar()
        # plt.savefig(f'./Figure/inter.png')
        x = self.decoder(x)
        
        return x