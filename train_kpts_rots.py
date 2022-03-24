import pickle
import numpy as np
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from config import *
from src.model_kpt_rot import SixDOFNet
from src.food_kpt_rot_dataset import PoseDataset, transform

kpt_loss_weight = 0.99995
rot_loss_weight = 0.00005

os.environ["CUDA_VISIBLE_DEVICES"]="0"

PI = torch.Tensor([np.pi]).double().cuda()

def rad2deg(rad):
    return rad*(180/PI)

def angle_loss(a,b):
     #loss = MSE(torch.rad2deg(a), torch.rad2deg(b))
     loss = MSE(rad2deg(a), rad2deg(b))
     return loss

def forward(sample_batched, model):
    img, img_np, gt_gauss, gt_rot = sample_batched
    img = Variable(img.cuda() if use_cuda else img)
    pred_gauss, pred_rot = model.forward(img)
    rot_loss = angle_loss(gt_rot.squeeze(), pred_rot.double().squeeze())
    kpt_loss = bceLoss(pred_gauss.double(), gt_gauss)
    return rot_loss_weight*rot_loss, kpt_loss_weight*kpt_loss

def fit(train_data, test_data, model, epochs, checkpoint_path = ''):
    for epoch in range(epochs):

        train_loss = 0.0
        train_kpt_loss = 0.0
        train_rot_loss = 0.0
        for i_batch, sample_batched in enumerate(train_data):
            optimizer_kpt.zero_grad()
            optimizer_rot.zero_grad()
            rot_loss, kpt_loss = forward(sample_batched, model)
            rot_loss.backward(retain_graph=True)
            kpt_loss.backward(retain_graph=True)
            optimizer_rot.step()
            optimizer_kpt.step()
            train_loss += kpt_loss.item() + rot_loss.item()
            train_kpt_loss += kpt_loss.item()
            train_rot_loss += rot_loss.item()
            print('[%d, %5d] kpts loss: %.3f, rot loss: %.3f' % (epoch + 1, i_batch + 1, kpt_loss.item(), rot_loss.item()), end='')
            print('\r', end='')
        print('train kpt loss:', (1/kpt_loss_weight)*train_kpt_loss/i_batch)
        print('train rot loss:', np.sqrt((1/rot_loss_weight)*(train_rot_loss/i_batch)))
        
        test_loss = 0.0
        test_kpt_loss = 0.0
        test_rot_loss = 0.0
        for i_batch, sample_batched in enumerate(test_data):
            rot_loss, kpt_loss = forward(sample_batched, model)
            test_loss += kpt_loss.item() + rot_loss.item()
            test_kpt_loss += kpt_loss.item()
            test_rot_loss += rot_loss.item()
        print('test kpt loss:', (1/kpt_loss_weight)*test_kpt_loss/i_batch)
        print('test rot loss:', np.sqrt((1/rot_loss_weight)*test_rot_loss/i_batch))
        torch.save(model.state_dict(), checkpoint_path + '/model_2_1_' + str(epoch) + '.pth')

# dataset
workers=0
dataset_dir = 'acquis_rot_recenter'
output_dir = 'checkpoints'
save_dir = os.path.join(output_dir, dataset_dir, '_new')

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


train_dataset = PoseDataset('/host/datasets/%s/train'%dataset_dir, transform)
train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
test_dataset = PoseDataset('/host/datasets/%s/test'%dataset_dir, transform)
test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
use_cuda = torch.cuda.is_available()

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
if use_cuda:
    torch.cuda.set_device(0)

# model
model = SixDOFNet().cuda()

# optimizer
optimizer_kpt = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1.0e-4)
optimizer_rot = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1.0e-4)

MSE = torch.nn.MSELoss()
bceLoss = nn.BCELoss()

fit(train_data, test_data, model, epochs=epochs, checkpoint_path=save_dir)
