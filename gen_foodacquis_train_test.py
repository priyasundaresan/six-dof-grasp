import numpy  as np
import cv2
from random import shuffle
import os

def pad_img(img):
    """
    :param images: sequence of images
    :return: list of images padded so that all images have same width and height (max width and height are used)
    """
    h, w = img.shape[:2]
    width_max = max(h,w)
    height_max = max(h,w)

    h, w = img.shape[:2]
    diff_vert = height_max - h
    pad_top = diff_vert//2
    pad_bottom = diff_vert - pad_top
    diff_hori = width_max - w
    pad_left = diff_hori//2
    pad_right = diff_hori - pad_left
    img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)
    return img_padded

if __name__ == '__main__':
    dataset_dir = os.getcwd()
    output_train_dir = os.path.join(dataset_dir, 'train')
    output_train_img_dir = os.path.join(output_train_dir, 'images')
    output_val_dir = os.path.join(dataset_dir, 'test')
    output_val_img_dir = os.path.join(output_val_dir, 'images')
    dirs = [output_train_dir, output_train_img_dir, output_val_dir, output_val_img_dir]
    for d in dirs:
        if not os.path.exists(d):
            os.mkdir(d)

    listOfFiles = list()
    inp_img_dir = 'food_dset_recentered'
    for fn in os.listdir(inp_img_dir):
        if 'png' in fn:
            listOfFiles.append(os.path.join(inp_img_dir, fn))

    n_total_images = len(os.listdir(inp_img_dir))

    train_idx = 0
    val_idx = 0
    split_idx = int(n_total_images*0.8)
    print(split_idx)

    shuffle(listOfFiles)
    for idx, f in enumerate(listOfFiles):
        img = cv2.imread(f)
        #img = pad_img(img) 
        img = cv2.resize(img, (136,136))
        if idx < split_idx:
            cv2.imwrite(os.path.join(output_train_img_dir, '%05d.jpg'%train_idx), img)
            train_idx += 1
        else:
            cv2.imwrite(os.path.join(output_val_img_dir, '%05d.jpg'%val_idx), img)
            val_idx += 1
        if idx == n_total_images:
            break
