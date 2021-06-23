# LOKI: Local Oriented Knot Inspection

* This repo provides a lightweight, general framework for supervised learning of cable grasping from synthetic images. We use it to train LOKI: Local Oriented Knot Inspection, which plans grasping on cables given local image crops. For more details, see:
#### ["Untangling Dense Non-Planar Knots by Learning Manipulation Features and Recovery Policies"](https://sites.google.com/berkeley.edu/non-planar-untangling)
#### Priya Sundaresan*, Jennifer Grannen*, Brijen Thananjeyan, Ashwin Balakrishna, Jeffrey Ichnowski, Ellen Novoseller,  Minho Hwang, Michael Laskey, Joseph E. Gonzalez, Ken Goldberg (*equal contribution)

### Description
  * `docker`: Contains utils for building Docker images and Docker containers to manage Python dependencies used in this project 
  * `src`: Contains model definitions, dataloaders, and visualization utils
  * `config.py`: Script for configuring hyperparameters for a training job
  * `train.py`: Script for training
  * `analysis.py`: Script for running inference on a trained model and saving predicted keypoint visualizations


### Getting Started/Overview
#### Dataset Generation
* [six-dof-rendering](https://github.com/priyasundaresan/six-dof-rendering) is the repo for LOKI rendering/dataset gen, and this repo (particularly the branch kpt_rot) is for LOKI training 
* Clone [six-dof-rendering](https://github.com/priyasundaresan/six-dof-rendering), best done locally rather than on a remote host, since we use Blender rendering which has not been tested headless or with X11 forwarding/VirtualGL.
* For `six-dof-rendering`, you need to install Blender (download instructions here https://github.com/priyasundaresan/blender-rope-sim#setup), and then the usage is as follows:
* `blender -b -P render.py`
* This will yield two folders, `images` and `annots` at the same level as `render.py` which contains synthetic cable crops and the associated grasp label (offset and orientation)
* Run `python real_kp_augment.py` to data augment using affine transformations and domain randomization
* This produces a folder called `aug` containing `images` and `annots`
```
|-- aug
|   |-- images
|   `-- annots
```
* `cp -r aug/* . `
* `python vis.py` which will show the rendered images & ground truth state annotations to a directory called `vis`, which will produce visualizations `vis/00000.jpg, vis/00001.jpg, ... `
* For making a train dataset, you'll repeat the above steps rendering a roughly ~80-20 split (augmented to 3000, 750 train / test images) and rename the `aug` folders to `dset_train`, `dset_test` so that each contains `dset_train/images`, `dset_train/annots`, `dset_test/images`, `dset_test/images`
#### Getting Started with Training/Inference
* Clone this repo with `https://github.com/priyasundaresan/six-dof-grasp/tree/kpt_rot` and switch to the `kpt_rot` branch with `git checkout --track origin/kpt_rot`
* Run `cd docker` and then `./docker_build.py`: This is a one-time step to build a Docker image called `priya-keypoints` according to the dependencies in `docker/Dockerfile`. Please note that this step may take a few minutes to run all the necessary installs.
* Check that the Docker image was built by running `docker images`, which should display a line like so:
```
priya-keypoints                      latest                          735686b1cd81        2 months ago        5.17GB
```
* Check that you can launch a Docker container from the image you created; `cd docker` and run `./docker_run.py` which should open a container with a prompt like:
```
root@afc66cb0930c:/host#
```
* Note that `Ctrl + D` allows you to detach out of the container at any time and all development should happen while in the container
* While in the container, make a folder called `datasets` at the same level as `src`, `docker`, `vis` (this will store all future datasets)
* Copy your locally rendered dataset from above (Copy the locally rendered `dset_train` and `dset_test` to `datasets/`
* The final structure should be as follows:
```
datasets/
|--dset_train/
|   |-- images
|   |   `-- 00000.jpg
|   |   ...
|   `-- annots
|       `-- 00000.npy
|       ...
|--dset_test/
    |-- images
    |   `-- 00000.jpg
    |   ...
    `-- annots
        `-- 00000.npy
        ...
```
* Configure https://github.com/priyasundaresan/six-dof-grasp/blob/76b5d74d929f0d22af1725e4eca246aaa5767c97/train.py#L84 and https://github.com/priyasundaresan/six-dof-grasp/blob/76b5d74d929f0d22af1725e4eca246aaa5767c97/train.py#L86 with `dset_train`  and `dset_test`
* Run `python train.py`
* Put your saved checkpoint in `analysis.py` and run `python analysis.py` which will produce visualizations on the real test crops here: https://github.com/priyasundaresan/six-dof-grasp/tree/kpt_rot/datasets/crops

### Contributing 
* For any questions, contact [Priya Sundaresan](http://priya.sundaresan.us) at priya.sundaresan@berkeley.edu or [Jennifer Grannen](http://jenngrannen.com/) at jenngrannen@berkeley.edu, or alternatively feel free to file an issue report and we'll get back to you ASAP!
* Happy rendering & grasping :) 
