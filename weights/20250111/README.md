## 训练
```
python -m modules.training.train_diy --training_type xfeat_default --megadepth_root_path /home/datashare/accelerated_features/megadepth  --synthetic_root_path /home/datashare/accelerated_features/coco_20k --ckpt_save_path ckpts/ --batch_size 50 --device_num 1
```

## 评估
### megadepth1500
```
python3 -m modules.eval.megadepth1500_diy --dataset-dir dataset/Mega1500/ --matcher xfeat --ransac-thr 2.5 --weights ckpts/xfeat_default_158500.pth
```
结果
```
auc@5 :  43.4
auc@10 :  57.4
auc@20 :  68.8
mAcc@5: 65.1
mAcc@10: 76.1
mAcc@20: 83.0
```

### scannet1500
```
python3 -m modules.eval.scannet1500_diy --scannet_path dataset/ScanNet1500 --output dataset/ScanNet1500/output --weights ckpts/xfeat_default_158500.pth --force && python3 -m modules.eval.scannet1500_diy --scannet_path dataset/ScanNet1500 --output dataset/ScanNet1500/output --show --weights ckpts/xfeat_default_158500.pth
```
结果
```
Dataset:scannet                         
Sorting by mean
         name best_thresh     5    10    20  mean
0       alike         4.5   7.9  16.6  26.2  16.9
1  xfeat_star         6.0  16.2  31.8  47.3  31.8
2       xfeat         4.0  17.5  33.1  48.3  32.9
```