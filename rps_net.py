'''
RPS network script with resnet-18
Copyright (c) Jathushan Rajasegaran, 2019
'''
from __future__ import print_function

# import argparse
# import os
# import shutil
# import torch
# import pdb
# import torch.nn as nn
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.optim as optim
# import torch.utils.data as data
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import copy
# import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
# from torch.autograd import Variable
# from torch.autograd import gradcheck


class RPS_net_cifar(nn.Module):

        def __init__(self, args):
            super(RPS_net_cifar, self).__init__()
            self.args = args
            self.final_layers = []
            self.init(None)

        def init(self, best_path):


            """Initialize all parameters"""
            self.conv1 = []
            self.conv2 = []
            self.conv3 = []
            self.conv4 = []
            self.conv5 = []
            self.conv6 = []
            self.conv7 = []
            self.conv8 = []
            self.conv9 = []
            self.fc1 = []

            div = 1
            a1 = 64//div

            a2 = 64//div
            a3 = 128//div
            a4 = 256//div
            a5 = 512//div

            self.a5 =a5
            # conv1
            for i in range(self.args.M):
                exec("self.m1" + str(i) + " = nn.Sequential(nn.Conv2d(3, "+str(a1)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a1)+"),nn.ReLU())")
                exec("self.conv1.append(self.m1" + str(i) + ")")


            # conv2
            for i in range(self.args.M):
                exec("self.m2" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a1)+", "+str(a2)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a2)+"),nn.ReLU(), nn.Conv2d("+str(a2)+", "+str(a2)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a1)+"))")
                exec("self.conv2.append(self.m2" + str(i) + ")")
            

            # conv3
            for i in range(self.args.M):
                exec("self.m3" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a2)+", "+str(a2)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a2)+"),nn.ReLU(), nn.Conv2d("+str(a2)+", "+str(a2)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a2)+"))")
                exec("self.conv3.append(self.m3" + str(i) + ")")
           


            # conv4
            for i in range(self.args.M):
                exec("self.m4" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a2)+", "+str(a3)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a3)+"),nn.ReLU(), nn.Conv2d("+str(a3)+", "+str(a3)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a3)+"))")
                exec("self.conv4.append(self.m4" + str(i) + ")")
            

            # conv5
            for i in range(self.args.M):
                exec("self.m5" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a3)+", "+str(a3)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a3)+"),nn.ReLU(), nn.Conv2d("+str(a3)+", "+str(a3)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a3)+"))")
                exec("self.conv5.append(self.m5" + str(i) + ")")
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
           


            # conv6
            for i in range(self.args.M):
                exec("self.m6" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a3)+", "+str(a4)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a4)+"),nn.ReLU(), nn.Conv2d("+str(a4)+", "+str(a4)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a4)+"))")
                exec("self.conv6.append(self.m6" + str(i) + ")")
          

            # conv7
            for i in range(self.args.M):
                exec("self.m7" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a4)+", "+str(a4)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a4)+"),nn.ReLU(), nn.Conv2d("+str(a4)+", "+str(a4)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a4)+"))")
                exec("self.conv7.append(self.m7" + str(i) + ")")
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        

            # conv8
            for i in range(self.args.M):
                exec("self.m8" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a4)+", "+str(a5)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a5)+"),nn.ReLU(), nn.Conv2d("+str(a5)+", "+str(a5)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a5)+"))")
                exec("self.conv8.append(self.m8" + str(i) + ")")


            # conv9
            for i in range(self.args.M):
                exec("self.m9" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a5)+", "+str(a5)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a5)+"),nn.ReLU(), nn.Conv2d("+str(a5)+", "+str(a5)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a5)+"))")
                exec("self.conv9.append(self.m9" + str(i) + ")")
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

            if len(self.final_layers) < 1:
                self.final_layer1 = nn.Linear(a5, 100)
                self.final_layers.append(self.final_layer1)

            self.cuda()

        #def forward(self, x, path, last):
        def forward(self, x, path, last):


            #print("LAST VALUE:",last)
            y = 0
            for j in range(self.args.M):
                if(path[0][j]==1):
                    y += self.conv1[j](x)
            x = y
            x = F.relu(x)


            y = 0
            for j in range(self.args.M):
                if(path[1][j]==1):
                    y += self.conv2[j](x)
            x = y + x
            x = F.relu(x)


            y = 0
            for j in range(self.args.M):
                #print("args.M:",self.args.M)
                #print("Path:",path)
                #print("iter:",j)
                if(path[2][j]==1):
                #if(path[0][j]==1):
                    y += self.conv3[j](x)
            #print("Yyyyyyy:",y)
            x = y + x
            x = F.relu(x)


            y = 0
            for j in range(self.args.M):
                if(path[3][j]==1):
                    y += self.conv4[j](x)
                    #print("Yyyyyyy:",y)
                #if (x == 0):
                    #break
            
            x = y
            #print("Xxxxxx:", x)
            #print("4th Iteration",x)
            x = F.relu(x)

            y = 0
            for j in range(self.args.M):
                if(path[4][j]==1):
                    y += self.conv5[j](x)
            x = y + x
            x = F.relu(x)
            x = self.pool1(x)


            y = 0
            for j in range(self.args.M):
                if(path[5][j]==1):
                    y += self.conv6[j](x)
            x = y
            x = F.relu(x)

            y = 0
            for j in range(self.args.M):
                if(path[6][j]==1):
                    y += self.conv7[j](x)
            x = y + x
            x = F.relu(x)
            x = self.pool2(x)

            y = 0
            for j in range(self.args.M):
                if(path[7][j]==1):
                    y += self.conv8[j](x)
            x = y
            x = F.relu(x)

            y = 0
            for j in range(self.args.M):
                if(path[8][j]==1):
                    y += self.conv9[j](x)
            x = y + x
            x = F.relu(x)
#             x = self.pool3(x)

            x = F.avg_pool2d(x, (8, 8), stride=(1,1))
            x = x.view(-1, self.a5)
            x = self.final_layers[last](x)

            return x
