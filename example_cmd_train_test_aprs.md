# Training and Testing single scene APRs

## PoseNet + EfficientNet
### Train
```
main.py posenet train models/backbones/efficient-net-b0.pth
/mnt/data/CambridgeLandmarks/CAMBRIDGE_dataset/
datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_KingsCollege_train.csv
CambridgeLandmarks_config.json
```
### Test
```
main.py 
posenet
test
models/backbones/efficient-net-b0.pth
/mnt/data/CambridgeLandmarks/CAMBRIDGE_dataset/
datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_KingsCollege_test.csv
CambridgeLandmarks_config.json
--checkpoint_path
posenet_effnet_apr_kings_college.pth
```

## PoseNet + ResNet50
### Train
```
main.py posenet train resnet50_pretrained.pth
/mnt/data/CambridgeLandmarks/CAMBRIDGE_dataset/
datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_KingsCollege_train.csv
CambridgeLandmarks_config.json
```
### Test
```
main.py 
posenet
test
resnet50_pretrained.pth
/mnt/data/CambridgeLandmarks/CAMBRIDGE_dataset/
datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_KingsCollege_test.csv
CambridgeLandmarks_config.json
--checkpoint_path
posenet_resnet50_apr_kings_college.pth
```

## MS-Transposenet 
See detailed examples at: https://github.com/yolish/multi-scene-pose-transformer
