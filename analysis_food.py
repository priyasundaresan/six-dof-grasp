import cv2
import os
import cmath
import math
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from config import *
from src.model import SixDOFNet
from src.food_dataset import PoseDataset, transform
import numpy as np
from scipy.spatial.transform import Rotation as R

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def expectation(d):
    width, height = d.T.shape
    d = d.T.ravel()
    #print(np.amin(d), np.amax(d))
    d[np.where(d < 0.15)] = 0.0
    d_norm = d/d.sum()
    x_indices = np.array([i % width for i in range(width*height)])
    y_indices = np.array([i // width for i in range(width*height)])
    exp_val = [int(np.dot(d_norm, x_indices)), int(np.dot(d_norm, y_indices))]
    return exp_val

def run_inference(model, img, output_dir='vis'):
    img_t = transform(img)
    img_t = img_t.cuda().unsqueeze(0)
    H,W,C = img.shape
    render_size = (W,H)
    heatmap, pred = model(img_t)
    heatmap = heatmap.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy().squeeze()
    heatmap = heatmap[0][0]
    #pred_x, pred_y = expectation(heatmap)
    pred_y, pred_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
    pred_y, pred_x = H//2, W//2
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.addWeighted(img, 0.55, heatmap, 0.45, 0)
    cv2.circle(heatmap, (pred_x,pred_y), 2, (255,255,255), -1)
    cv2.circle(heatmap, (W//2,H//2), 2, (0,0,0), -1)
    pt = cmath.rect(20, np.pi/2-pred)  
    x2 = int(pt.real)
    y2 = int(pt.imag)
    rot_vis = cv2.line(img, (pred_x-x2,pred_y+y2), (pred_x+x2, pred_y-y2), (255,255,255), 2)
    cv2.putText(heatmap,"Skewer Point",(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    cv2.putText(rot_vis,"Skewer Axis",(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    cv2.circle(rot_vis, (pred_x,pred_y), 4, (255,255,255), -1)
    result = np.hstack((heatmap, rot_vis))
    return math.degrees(pred), result

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    model = SixDOFNet()
    model.load_state_dict(torch.load('/host/checkpoints/acquis_3kpt_clean/model_2_1_19.pth'))
    #model.load_state_dict(torch.load('/host/checkpoints/spaghetti_rot_recenter/model_2_1_19.pth'))
    torch.cuda.set_device(0)
    model = model.cuda()
    model.eval()
    output_dir = 'vis'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    #test_dir = 'crops_test'
    test_dir = 'test_images'
    for idx, f in enumerate(sorted(os.listdir(test_dir))):
        img = cv2.imread(os.path.join(test_dir, f))
        angle, vis = run_inference(model, img, output_dir)
        print(idx, angle)
        #print("Annotating %06d"%idx)
        annotated_filename = "%05d.jpg"%idx
        cv2.imwrite('%s/%s'%(output_dir, annotated_filename), vis)

    #dataset_dir = 'spaghetti_rot_recenter'
    #test_dataset = PoseDataset('/host/datasets/%s/train'%dataset_dir, transform)
    ##test_dataset = PoseDataset('/host/datasets/%s/test'%dataset_dir, transform)
    #for i in range(len(test_dataset)):
    #    img, img_np, gauss, labels = test_dataset[i]
    #    gt_angle = np.degrees(labels.cpu().numpy())
    #    angle, vis = run_inference(model, img_np, output_dir)
    #    print(i, angle, gt_angle)
    #    annotated_filename = "%05d.jpg"%i
    #    cv2.imwrite('%s/%s'%(output_dir, annotated_filename), vis)
