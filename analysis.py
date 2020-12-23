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

def draw(img, source_px, imgpts, intensity=255):
    imgpts = imgpts.astype(int)
    img = cv2.arrowedLine(img, source_px, tuple(imgpts[0].ravel()), (intensity,0,0), 2)
    img = cv2.arrowedLine(img, source_px, tuple(imgpts[1].ravel()), (0,intensity,0), 2)
    img = cv2.arrowedLine(img, source_px, tuple(imgpts[2].ravel()), (0,0,intensity), 2)
    return img 

def project_3d_point(transformation_matrix,p,render_size):
    p1 = transformation_matrix @ Vector((p.x, p.y, p.z, 1)) 
    p2 = Vector(((p1.x/p1.w, p1.y/p1.w)))
    p2 = (np.array(p2) - (-1))/(1 - (-1)) # Normalize -1,1 to 0,1 range
    pixel = [int(p2[0] * render_size[0]), int(render_size[1] - p2[1]*render_size[1])]
    return pixel

def proj_axes_from_trans_rot(trans, rot_euler, render_size):
    axes = np.float32([[1,0,0],[0,1,0],[0,0,-1]])*0.2
    rot_mat = R.from_euler('xyz', rot_euler).as_matrix()
    axes = rot_mat@axes
    axes += trans
    axes_projected = []
    for axis in axes:
        axes_projected.append(project_3d_point(world_to_cam, Vector(axis), render_size))
    axes_projected = np.array(axes_projected)
    center_projected = project_3d_point(world_to_cam, Vector(trans), render_size)
    center_projected = tuple(center_projected)
    return center_projected, axes_projected

def run_inference(model, img, world_to_cam, gt_rot=None, output_dir='vis'):
    img_t = transform(img)
    img_t = img_t.cuda().unsqueeze(0)
    H,W,C = img.shape
    render_size = (W,H)
    pred = model(img_t).detach().cpu().numpy().squeeze()
    trans = trans_gt = np.zeros(3)
    rot_euler = np.array([0,0,pred])
    #rot_euler = pred
    center_projected_pred, axes_projected_pred = proj_axes_from_trans_rot(trans_gt, rot_euler, render_size)
    vis_pred = draw(img.copy(),center_projected_pred,axes_projected_pred)
    cv2.putText(vis_pred,"Pred",(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    if gt_rot is not None:
        center_projected_gt, axes_projected_gt = proj_axes_from_trans_rot(trans_gt, rot_euler_gt, render_size)
        vis_gt = draw(img.copy(),center_projected_gt,axes_projected_gt, intensity=100)
        cv2.putText(vis_gt,"Ground Truth",(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        result = np.hstack((vis_gt, vis_pred))
    else:
        result = vis_pred
    return result

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    model = SixDOFNet()
    #model.load_state_dict(torch.load('/host/checkpoints/cyl/model_2_1_17.pth'))
    model.load_state_dict(torch.load('/host/checkpoints/more_blur_1rot/model_2_1_29.pth'))
    torch.cuda.set_device(0)
    model = model.cuda()
    dataset_dir = '/host/datasets/more_blur/test'
    #dataset_dir = '/host/datasets/cyl_test'
    image_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'annots')
    world_to_cam = Matrix(np.load('%s/cam_to_world.npy'%(labels_dir)))
    output_dir = 'vis'
#    test_dir = '/host/datasets/more_blur/test/images'
    test_dir = '/host/datasets/crops'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    #for idx, f in enumerate(sorted(os.listdir(image_dir))):
    #    img = cv2.imread(os.path.join(image_dir, f))
    #    img = cv2.resize(img, (200,200))
    #    gt_data = np.load(os.path.join(labels_dir, '%05d.npy'%idx), allow_pickle=True)
    #    gt_trans = gt_data.item().get("trans")
    #    gt_rot = gt_data.item().get("rot")
    #    gt_label = np.hstack((gt_trans, gt_rot))
    #    vis = run_inference(model, img, gt_label, world_to_cam, output_dir)
    #    print("Annotating %06d"%idx)
    #    annotated_filename = "%05d.jpg"%idx
    #    cv2.imwrite('%s/%s'%(output_dir, annotated_filename), vis)
    #    if idx > 25:
    #        break
    for idx, f in enumerate(sorted(os.listdir(test_dir))):
        img = cv2.imread(os.path.join(test_dir, f))
        print(os.path.join(test_dir, f))
        try:
            img = cv2.resize(img, (200,200))
            gt_trans = np.zeros(3)
            gt_rot = None
            vis = run_inference(model, img, world_to_cam, gt_rot, output_dir)
            print("Annotating %06d"%idx)
            annotated_filename = "%05d.jpg"%idx
            cv2.imwrite('%s/%s'%(output_dir, annotated_filename), vis)
        except:
            pass
        if idx > 200:
            break
