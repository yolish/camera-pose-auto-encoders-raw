## Learning Pose Auto-Encoders for Camera Pose Regrssion
Official PyTorch implementation of pose auto-encoders for camera pose regression.
APR architectures and training were cloned from: https://github.com/yolish/multi-scene-pose-transformer

### Repository Overview 

This code implements:

1. Training and testing of single and multi-scene APRs: PoseNet with different backbones and MS-Transformer
2. Training and testing of pose auto-encoders
3. Training and testing of APRs + pose auto-encoders -based refinement 

### Prerequisites

In order to run this repository you will need:

1. Python3 (tested with Python 3.7.7)
1. PyTorch deep learning framework (tested with version 1.0.0)
1. Use torch==1.4.0, torchvision==0.5.0
1. Download the [Cambridge Landmarks](http://mi.eng.cam.ac.uk/projects/relocalisation/#dataset) dataset and the [7Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) dataset
1. You an also download pre-trained models to reproduce reported results (see below)
Note: All experiments reported in our paper were performed with an 8GB 1080 NVIDIA GeForce GTX GPU

### Usage
#### Training and testing single- and multi-scene APRs
The entry point for training and testing is the ```main.py``` script in the root directory
 For detailed explanation of the options run:
  ```
  python main.py -h
  ```
See ```example_cmd_train_test_aprs.md``` for detailed examples of how to run this script.
#### Training and testing Camera Pose Auto-Encoders
The entry point for training and testing are the ```main_learn_pose_encoding.py``` and ```main_learn_multiscene_pose_encoding.py.py``` scripts in the root directory
for training auto-encoders for single and multi-scene APRs, respectively. 

See ```example_cmd_train_test_pose_auto_encoders.md``` for detailed examples of how to run these scripts.

#### Training and testing APRs with auto-encoders priors
The entry point for training and testing is the ```main_refine_apr.py``` script.

See ```example_cmd_train_test_apr_with_auto_encoders_priors.md``` for detailed examples of how to run this script.

### Pre-trained models
You can download pretrained models in order to easily reproduce our results:

|Model|Description|Train/Test Script|
|---|---|---|
|APRs|
PoseNet+EfficientNetB0|Single Scene APR: KingsCollege scene|main.py|
PoseNet+ResNet50|Single Scene APR: KingsCollege scene|main.py|
PoseNet+MobileNet v2|Single Scene APR: KingsCollege scene|main.py|
MSTransformer|Multi-Scene APR: CambridgeLandmarks|main.py, see also: https://github.com/yolish/multi-scene-pose-transformer |
MSTransformer|Multi-Scene APR: 7Scenes|main.py, see also: https://github.com/yolish/multi-scene-pose-transformer |
|Auto-Encoderes|
Pose Auto-Encoder for PoseNet+EfficientNetB0|AutoEncoder: KingsCollege scene|main_learn_pose_encoding.py|
Pose Auto-Encoder for PoseNet+ResNet50|AutoEncoder: KingsCollege scene|main_learn_pose_encoding.py|
Pose Auto-Encoder for PoseNet+MobileNet v2|AutoEncoder: KingsCollege scene|main_learn_pose_encoding.py|
Pose Auto-Encoder for MSTransformer|AutoEncoder: CambridgeLandmarks|main_learn_multiscene_pose_encoding.py.py|
Pose Auto-Encoder for MSTransformer|AutoEncoder: 7Scenes|main_learn_multiscene_pose_encoding.py.py|

 
  
  
  
  
