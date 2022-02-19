## Camera Pose Auto-Encoders
Official PyTorch implementation of pose auto-encoders for camera pose regression.

### Repository Overview 

This code implements:

1. Training and testing of single and multi-scene APRs: PoseNet with different backbones and MS-Transformer. MS-Transformer and its training/training (1) was cloned from: https://github.com/yolish/multi-scene-pose-transformer
2. Training and testing of pose auto-encoders
3. Test-time optimization for position regression with camera pose encoding
4. Image reconstructiom from camera pose encoding

### Prerequisites

In order to run this repository you will need:

1. Python3 (tested with Python 3.7.7)
1. PyTorch deep learning framework (tested with version 1.0.0)
1. Use torch==1.4.0, torchvision==0.5.0
1. Download the [Cambridge Landmarks](http://mi.eng.cam.ac.uk/projects/relocalisation/#dataset) dataset and the [7Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) dataset
1. You an also download pre-trained models to reproduce reported results (see below)
Note: All experiments reported in our paper were performed with an 8GB 1080 NVIDIA GeForce GTX GPU
1. For a quick set up you can run: pip install -r requirments.txt 

### Usage
#### Training and testing single- and multi-scene APRs
The entry point for training and testing APRs is the ```main.py``` script in the root directory
See ```example_cmd/example_cmd_train_test_aprs.md``` for example command lines.

#### Training and testing Camera Pose Auto-Encoders
The entry point for training and testing camera pose auto-encoders are the ```main_learn_pose_encoding.py``` and ```main_learn_multiscene_pose_encoding.py.py``` scripts in the root directory
corresponding to auto-encoders for single and multi-scene APRs. 
See examle_cmd\example_cmd_train_test_pose_auto_encoders.md for example command lines.

#### Test-time optimization for position regression with camera pose encoding
The entry point for test time optimization is the ```main_refine_apr_test_time.py``` script.
See ```example_cmd\example_cmd_test_time_optim.md``` for example command lines.

#### Image reconstructiom from camera pose encoding
The entry training and testing an image decoder to reconstruct images from camera pose encoding, is the ```main_reconstruct_img.py``` script.
See ```example_cmd\example_cmd_reconstruct_img.md``` for example command lines.


### Pre-trained models
You can download pretrained models in order to easily reproduce our results 



 
  
  
  
  
