import cv2
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from config import *
from src.model import SixDOFNet
from src.dataset import PoseDataset, transform
import numpy as np
from mathutils import *
from scipy.spatial.transform import Rotation as R

def draw(img, source_px, imgpts):
    imgpts = imgpts.astype(int)
    img = cv2.line(img, source_px, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, source_px, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, source_px, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img 

def project_3d_point(transformation_matrix,p,render_size):
    p1 = transformation_matrix @ Vector((p.x, p.y, p.z, 1)) 
    p2 = Vector(((p1.x/p1.w, p1.y/p1.w)))
    p2 = (np.array(p2) - (-1))/(1 - (-1)) # Normalize -1,1 to 0,1 range
    pixel = [int(p2[0] * render_size[0]), int(render_size[1] - p2[1]*render_size[1])]
    return pixel

def run_inference(model, img, world_to_cam, output_dir='vis'):
    img_t = transform(img)
    img_t = img_t.cuda().unsqueeze(0)
    H,W,C = img.shape
    render_size = (W,H)
    pred = model(img_t).detach().cpu().numpy().squeeze()
    trans = pred[:3]
    rot_euler = pred[3:]
    rot_mat = R.from_euler('xyz', rot_euler).as_matrix()
    axes = np.float32([[1,0,0],[0,1,0],[0,0,-1]])
    axes = rot_mat@axes
    axes += trans
    axes_projected = []
    center_projected = project_3d_point(world_to_cam, Vector(trans), render_size)
    for axis in axes:
        axes_projected.append(project_3d_point(world_to_cam, Vector(axis), render_size))
    axes_projected = np.array(axes_projected)
    center_projected = tuple(center_projected)
    vis = img.copy()
    vis = draw(vis,center_projected,axes_projected)
    return vis

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    model = SixDOFNet()
    model.load_state_dict(torch.load('/host/checkpoints/monkey/model_2_1_9.pth'))
    torch.cuda.set_device(0)
    model = model.cuda()
    dataset_dir = '/host/datasets/monkey_test'
    image_dir = os.path.join(dataset_dir, 'images')
    world_to_cam = Matrix(np.load('%s/annots/cam_to_world.npy'%dataset_dir))
    output_dir = 'vis'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for idx, f in enumerate(sorted(os.listdir(image_dir))):
        img = cv2.imread(os.path.join(image_dir, f))
        vis = run_inference(model, img, world_to_cam, output_dir)
        print("Annotating %06d"%idx)
        annotated_filename = "%05d.jpg"%idx
        cv2.imwrite('%s/%s'%(output_dir, annotated_filename), vis)
        if idx > 25:
            break
