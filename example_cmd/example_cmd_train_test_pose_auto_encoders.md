# Training and Testing Pose Auto Encoders

## Auto Encoder for PoseNet + EfficientNet (Single Scene)
### Train
```
main_learn_pose_encoding.py
posenet
train
models/backbones/efficient-net-b0.pth
/mnt/data/CambridgeLandmarks/CAMBRIDGE_dataset/
datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_KingsCollege_train.csv
CambridgeLandmarks_config.json
posenet_effnet_apr_kings_college.pth
```
###Test
```
main_learn_pose_encoding.py
posenet
test
models/backbones/efficient-net-b0.pth
/mnt/data/CambridgeLandmarks/CAMBRIDGE_dataset/
datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_KingsCollege_test.csv
CambridgeLandmarks_config.json
posenet_effnet_apr_kings_college.pth
--encoder_checkpoint_path
posenet_effnet_apr_kings_college.pth
```

## Auto Encoder for MS-Transformer (Multi-Scene)
### Train (Cambridge)
```
main_learn_multiscene_pose_encoding.py
ems-transposenet
train
/models/backbones/efficient-net-b0.pth
/mnt/data/CambridgeLandmarks/CAMBRIDGE_dataset/
datasets/CambridgeLandmarks/cambridge_four_scenes.csv
CambridgeLandmarks_config.json
ems_transposenet_cambridge_pretrained_finetuned.pth
```
### Train (7Scenes)
```
main_learn_multiscene_pose_encoding.py
ems-transposenet
train
models/backbones/efficient-net-b0.pth
/data/Datasets/7Scenes/
datasets/7Scenes/7scenes_all_scenes.csv
7Scenes_config.json 
ems_transposenet_7scenes_pretrained.pth
```

### Test - Example on Cambridge: Kings College
```
main_learn_multiscene_pose_encoding.py
ems-transposenet
train
/models/backbones/efficient-net-b0.pth
/mnt/data/CambridgeLandmarks/CAMBRIDGE_dataset/
datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_KingsCollege_test.csv
CambridgeLandmarks_config.json
ems_transposenet_cambridge_pretrained_finetuned.pth
--encoder_checkpoint_path 
mstransformer_cambridge_pose_encoder.pth
```

