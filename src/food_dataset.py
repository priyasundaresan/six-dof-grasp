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

def normalize(x):
    return F.normalize(x, p=1)

def gauss_2d_batch(width, height, sigma, U, V, normalize_dist=False):
    U.unsqueeze_(1).unsqueeze_(2)
    V.unsqueeze_(1).unsqueeze_(2)
    X,Y = torch.meshgrid([torch.arange(0., width), torch.arange(0., height)])
    X,Y = torch.transpose(X, 0, 1).cuda(), torch.transpose(Y, 0, 1).cuda()
    G=torch.exp(-((X-U.float())**2+(Y-V.float())**2)/(2.0*sigma**2))
    if normalize_dist:
        return normalize(G).double()
    return G.double()

def vis_gauss(img, gauss, title):
    gauss = gauss.squeeze(0).cpu().numpy()
    output = cv2.normalize(gauss, None, 0, 255, cv2.NORM_MINMAX)
    output = np.stack((output,)*3, axis=-1)
    added_image = cv2.addWeighted(img.astype(np.uint8),0.4,output.astype(np.uint8),0.6,0)
    cv2.imwrite(title, added_image)

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

class PoseDataset(Dataset):
	def __init__(self, dataset_dir, transform, img_height=136, img_width=136, gauss_sigma=5):
		self.transform = transform
		self.imgs = []
		self.img_height = img_height
		self.img_width = img_width
		self.gauss_sigma = gauss_sigma
		self.rots = []
		self.pixels = []
		labels_folder = os.path.join(dataset_dir, 'keypoints')
		img_folder = os.path.join(dataset_dir, 'images')
		for i in range(len(os.listdir(img_folder))-1):
                    self.imgs.append(os.path.join(img_folder, '%05d.jpg'%i))
                    annot = np.load(os.path.join(labels_folder, '%05d.npy'%i), allow_pickle=True)
                    label = annot[:-1]
                    center = np.reshape(annot[-1], (1,2))

                    center_coordinateaxes = center.copy()
                    center_coordinateaxes[:,1] = self.img_height - center_coordinateaxes[:,1]

                    label[:,1] = self.img_height - label[:,1] 
                    label -= center_coordinateaxes
                    angle = angle_between([-136,0], label[0])
                    rot = np.array(np.radians(angle))
                    center[:,0] = np.clip(center[:, 0], 0, self.img_width-1)
                    center[:,1] = np.clip(center[:, 1], 0, self.img_height-1)
                    self.rots.append(torch.from_numpy(rot).cuda())
                    self.pixels.append(torch.from_numpy(center).cuda())

	def __getitem__(self, index):  
		img_np = cv2.imread(self.imgs[index])
		img_np = cv2.resize(img_np, (self.img_width,self.img_height))
		img = self.transform(img_np)
		rot = self.rots[index]
		pixel = self.pixels[index]
		U = pixel[:,0]
		V = pixel[:,1]
		gaussian = gauss_2d_batch(self.img_width, self.img_height, self.gauss_sigma, U, V)
		return img, img_np, gaussian, rot
    
	def __len__(self):
		return len(self.rots)

if __name__ == '__main__':
	dset = PoseDataset('/host/datasets/acquis_rot2/train', transform)
	for i in range(50):
		img, img_np, gauss, labels = dset[i]
		print(i, np.degrees(labels.cpu().numpy()))
		vis_gauss(img_np, gauss, 'test%02d.png'%i)
