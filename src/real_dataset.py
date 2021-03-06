import torch
import cv2
import time
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, utils
import numpy as np
import pickle
import os
from datetime import datetime

transform = transforms.Compose([transforms.ToTensor()])

class PoseDataset(Dataset):
	def __init__(self, dataset_dir, transform):
		self.transform = transform
		self.imgs = []
		self.labels = []
		labels_folder = os.path.join(dataset_dir, 'annots')
		img_folder = os.path.join(dataset_dir, 'images')
		for i in range(len(os.listdir(img_folder))-1):
			self.imgs.append(os.path.join(img_folder, '%05d.jpg'%i))
			label = np.load(os.path.join(labels_folder, '%05d.npy'%i), allow_pickle=True)
			pose = np.array([label])
			self.labels.append(torch.from_numpy(pose).cuda())

	def __getitem__(self, index):  
		img_np = cv2.imread(self.imgs[index])
		img_np = cv2.resize(img_np, (200,200))
		img = self.transform(img_np)
		labels = self.labels[index]
		return img, labels
    
	def __len__(self):
		return len(self.labels)

if __name__ == '__main__':
	dset = PoseDataset('/host/datasets/cyl_dr_test', transform)
	img, labels = dset[0]
	print(img.shape, labels.shape)
