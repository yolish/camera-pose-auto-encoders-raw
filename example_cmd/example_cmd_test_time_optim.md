# Test-time optimization of position estimation with pose encoding
Optimization of position for MS-Transformer with a camera pose encoding (KingsCollege scene)
```
main_refine_apr_test_time.py
ems-transposenet
models/backbones/efficient-net-b0.pth
/mnt/data/CambridgeLandmarks/CAMBRIDGE_dataset/
datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_KingsCollege_test.csv
CambridgeLandmarks_config.json
pretrained_models/ems_transposenet_cambridge_pretrained_finetuned.pth
pretrained_models/mstransformer_cambridge_pose_encoder.pth
datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_KingsCollege_train.csv
```

Optimization of position for MS-Transformer with a camera pose encoding (Fire scene)
```
main_refine_apr_test_time.py
ems-transposenet
models/backbones/efficient-net-b0.pth
/data/Datasets/7Scenes/
datasets/7Scenes/abs_7scenes_pose.csv_fire_test.csv
7Scenes_config.json
pretrained_models/ems_transposenet_7scenes_pretrained.pth
pretrained_models/mstransformer_7scenes_pose_encoder.pth
datasets/7Scenes/abs_7scenes_pose.csv_fire_train.csv
```

## Optimization of position for PoseNet with a MobileNet backbone with a camera pose encoding (Kings College scene)
In CambridgeLandmarks_config.json change backbone_type to mobilenet
```
main_refine_apr_test_time.py
posenet
pretrained_models/mobilenet_v2_pretrained.pth
/mnt/data/CambridgeLandmarks/CAMBRIDGE_dataset/
datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_KingsCollege_test.csv
CambridgeLandmarks_config.json
pretrained_models/posenet_mobilenet_apr_kings_college.pth
pretrained_models/posenet_mobilenet_pose_encoder_kings_college.pth
datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_KingsCollege_train.csv
```
## Optimization of position for PoseNet with a ResNet50 backbone with a camera pose encoding (Kings College scene)
In CambridgeLandmarks_config.json change backbone_type to resnet50
```
main_refine_apr_test_time.py
posenet
pretrained_models/resnet50_pretrained.pth
/mnt/data/CambridgeLandmarks/CAMBRIDGE_dataset/
datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_KingsCollege_test.csv
CambridgeLandmarks_config.json
pretrained_models/posenet_resnet50_apr_kings_college.pth
pretrained_models/posenet_resnet50_auto_encoder_kings_college.pth
datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_KingsCollege_train.csv
```

## Optimization of position for PoseNet with an EfficientNet backbone with a camera pose encoding (Kings College scene)
In CambridgeLandmarks_config.json change backbone_type to efficientnet
```
main_refine_apr_test_time.py
posenet
models/backbones/efficient-net-b0.pth
/mnt/data/CambridgeLandmarks/CAMBRIDGE_dataset/
datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_KingsCollege_test.csv
CambridgeLandmarks_config.json
pretrained_models/posenet_effnet_apr_kings_college.pth
pretrained_models/posenet_effnet_pose_encoder_kings_college.pth
datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_KingsCollege_train.csv
```