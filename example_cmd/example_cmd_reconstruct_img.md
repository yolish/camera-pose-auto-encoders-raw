# Image reconstruction from pose encoding

## Training a decoder on train pose auto-encoding
```
main_reconstruct_img.py
train
/mnt/data/CambridgeLandmarks/CAMBRIDGE_dataset/
datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_ShopFacade_train.csv
reconstruct_config.json
pretrained_models/mstransformer_cambridge_pose_encoder.pth
```

## Demo of reconstructoin 
```
main_reconstruct_img.py
demo
/mnt/data/CambridgeLandmarks/CAMBRIDGE_dataset/
datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_ShopFacade_train.csv
reconstruct_config.json
pretrained_models/mstransformer_cambridge_pose_encoder.pth
--decoder_checkpoint_path pretrained_models/img_decoder_shop_facade.pth
```
