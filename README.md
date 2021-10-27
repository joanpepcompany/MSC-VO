# MSC-VO

**Authors:** Joan P. Company-Corcoles, Emilio Garcia-Fidalgo and Alberto Ortiz

## Related Publications:

Exploiting Manhattan and Structural Constraints for Visual Odometry. [UnderReview]

## General considerations:

MSC-VO is a novel RGB-D visual odometer that increases the performance of the traditional point-based methods by combining points and lines as visual features. Furthermore, to increase the performance in low-textured environments, it leverages structural constraints and Manhattan Axes (MA) alignment in the local map optimization stage.  

# 1. License

MSC-VO is released under a [GPLv3 license](https://github.com/raulmur/ORB_SLAM2/blob/master/License-gpl.txt). For a list of all code/library dependencies (and associated licenses), please see [Dependencies_MSC-VO.md].

# 2. Prerequisites

## Pangolin
We use [Pangolin](https://github.com/stevenlovegrove/Pangolin) for visualization and user interface. Download and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.

## OpenCV
We use [OpenCV](http://opencv.org) to manipulate images and features. Download and install instructions can be found at: http://opencv.org. **Tested with OpenCV 3.2**.

## Eigen3
Required by g2o (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 3.1.0**.

## DBoW2 and g2o (Included in Thirdparty folder)
We use modified versions of the [DBoW2](https://github.com/dorian3d/DBoW2) library to perform place recognition and [g2o](https://github.com/RainerKuemmerle/g2o) library to perform non-linear optimizations. Both modified libraries (which are BSD) are included in the *Thirdparty* folder.

# 3. RGB-D Example

MSC-VO has been evaluated in the ICL-NUIM and the TUM datasets. Furthermore, it has been executed in custom datasets where images are obtained by a RealSense D435. 
  ```
  ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.bin Examples/RGB-D/DATASET_CONFIG.yaml PATH_TO_SEQUENCE_FOLDER ASSOCIATIONS_FILE BOOL_USE_VIEWER
  ```


## TUM Dataset

1. Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and uncompress it.

2. Associate RGB images and depth images using the python script [associate.py](http://vision.in.tum.de/data/datasets/rgbd-dataset/tools). We already provide associations for some of the sequences in *Examples/RGB-D/associations/*. You can generate your own associations file executing:

  ```
  python associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt
  ```

3. Execute the following command. Change `TUMX.yaml` to TUM1.yaml,TUM2.yaml or TUM3.yaml for freiburg1, freiburg2 and freiburg3 sequences respectively. Change `PATH_TO_SEQUENCE_FOLDER`to the uncompressed sequence folder. Change `ASSOCIATIONS_FILE` to the path to the corresponding associations file.

  ```
  ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.bin Examples/RGB-D/TUMX.yaml PATH_TO_SEQUENCE_FOLDER ASSOCIATIONS_FILE BOOL_USE_VIEWER
  ```
