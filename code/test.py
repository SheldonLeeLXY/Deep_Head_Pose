import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F


import datasets, hopenet, utils
from mtcnn import MTCNN
import cv2

from PIL import Image
# from google.colab.patches import cv2_imshow
import numpy as np



cudnn.enabled = True
gpu = 0
snapshot_path = "../hopenet_robust_alpha1.pkl"

# ResNet50 structure
model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)


# Load snapshot
# saved_state_dict = torch.load(snapshot_path)
saved_state_dict = torch.load(snapshot_path, map_location=torch.device('cpu'))
model.load_state_dict(saved_state_dict)



transformations = transforms.Compose([transforms.Resize(224),
transforms.CenterCrop(224), transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


# model.cuda(gpu)

# Ensure the model runs on the CPU
model.to(torch.device('cpu'))




# Test the Model
model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
total = 0

idx_tensor = [idx for idx in range(66)]
# idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
# Ensure the tensor is created on the CPU
idx_tensor = torch.FloatTensor(idx_tensor).to(torch.device('cpu'))


yaw_error = .0
pitch_error = .0
roll_error = .0

l1loss = torch.nn.L1Loss(size_average=False)
def test(path):
    img = Image.open(path)

    img1 = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    #assuming one face
    face=detector.detect_faces(img1)[0]['box']
    x,y,w,h=face
    #print(x,y,w,h)

    img=img.crop((int(x-20),int(y-20),int(x+w+20),int(y+h+20)))



    img = img.convert('RGB')

    cv2_img=np.asarray(img)
    #
    #print(cv2_img.shape)
    cv2_img=cv2.resize(cv2_img,(224,224))[:,:,::-1]
    cv2_img = cv2_img.astype(np.uint8).copy()
    img = transformations(img)

    img=img.unsqueeze(0)

    # images = Variable(img).cuda(gpu)
    images = Variable(img).to(torch.device('cpu'))

    yaw, pitch, roll = model(images)

    # Binned predictions
    _, yaw_bpred = torch.max(yaw.data, 1)
    _, pitch_bpred = torch.max(pitch.data, 1)
    _, roll_bpred = torch.max(roll.data, 1)

    # Continuous predictions
    yaw_predicted = utils.softmax_temperature(yaw.data, 1)
    pitch_predicted = utils.softmax_temperature(pitch.data, 1)
    roll_predicted = utils.softmax_temperature(roll.data, 1)

    yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
    pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
    roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99



    pitch = pitch_predicted[0]
    yaw = -yaw_predicted[0]
    roll = roll_predicted[0]
    print("pitch,yaw,roll",pitch,yaw,roll)
    utils.draw_axis(cv2_img, yaw_predicted[0], pitch_predicted[0], roll_predicted[0], size=100)
    cv2.imwrite('../res.jpg', cv2_img)

#put the path of your image here, result will be saved as /content/res.jpg
test("../a.jpg")