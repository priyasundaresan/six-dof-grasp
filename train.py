import pickle
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from config import *
from src.model import SixDOFNet
from src.dataset import PoseDataset, transform
MSE = torch.nn.MSELoss()
l1_loss = nn.L1Loss

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def forward(sample_batched, model):
    img, gt_labels = sample_batched
    img = Variable(img.cuda() if use_cuda else img)
    pred_labels = model.forward(img).double()
    loss = l1_loss()(pred_labels, gt_labels)
    return loss

def fit(train_data, test_data, model, epochs, checkpoint_path = ''):
    for epoch in range(epochs):

        train_loss = 0.0
        for i_batch, sample_batched in enumerate(train_data):
            optimizer.zero_grad()
            loss = forward(sample_batched, model)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i_batch + 1, loss.item()), end='')
            print('\r', end='')
        print('train loss:', train_loss / i_batch)
        
        test_loss = 0.0
        for i_batch, sample_batched in enumerate(test_data):
            loss = forward(sample_batched, model)
            test_loss += loss.item()
        print('test loss:', test_loss / i_batch)
        if epoch%1 == 0:
            torch.save(model.state_dict(), checkpoint_path + '/model_2_1_' + str(epoch) + '.pth')

# dataset
workers=0
dataset_dir = 'monkey'
output_dir = 'checkpoints'
save_dir = os.path.join(output_dir, dataset_dir)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

train_dataset = PoseDataset('/host/datasets/monkey_train', transform)
train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
test_dataset = PoseDataset('/host/datasets/monkey_test', transform)
test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

use_cuda = torch.cuda.is_available()

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
if use_cuda:
    torch.cuda.set_device(0)

# model
model = SixDOFNet().cuda()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=1.0e-4, weight_decay=1.0e-4)

fit(train_data, test_data, model, epochs=epochs, checkpoint_path=save_dir)
