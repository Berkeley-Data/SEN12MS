#!/bin/bash

export WANDB_PROJECT=scene_classification

for datasize in  1024 # 512 256 128 64
do
  for sensor_tp in s2 s1 s1s2 #rgb
  do
    for dataset in  sen12ms bigearthnet
    do
      for lr in 0.001 0.0001 0.00001
      do
        for epoch in 300
        do
          for label_tp in multi_label # single_label
          do
            for model in  Moco # Moco_1x1RND Moco
            do
              python classification/main_train.py --simple_scheme --use_fusion \
                                                  --lr ${lr} --use_lr_step --lr_step_size 30 --decay 1e-5 \
                                                  --batch_size 64 --num_workers 4 --data_size ${datasize} \
                                                  --dataset ${dataset} --label_split_dir splits \
                                                  --label_type ${label_tp} --output_pred \
                                                  --model ${model} --pt_dir pretrained/moco \
                                                  --epochs ${epoch} --eval \
                                                  --sensor_type ${sensor_tp} \
                                                  --exp_name full_fusion --pt_name stilted-mountain-91
            done
          done
        done
      done
    done
  done
done