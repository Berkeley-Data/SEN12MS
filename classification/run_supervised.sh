#!/bin/bash

export WANDB_PROJECT=sup_scene_cls

for datasize in 1024 # 128 64 512 256 1024
do
  for sensor_tp in  s2 # s1 s1s2 rgb
  do
    for dataset in sen12ms # bigearthnet #
    do
      for lr in 0.001 # 0.0001 0.00001
      do
        for epoch in 2 # 20 100 150 200 250 300
        do
          for label_tp in multi_label # single_label
          do
            for model in Supervised # Supervised_1x1
            do
              python classification/main_train.py --exp_name sup_learning \
                                                  --lr ${lr} --use_lr_step --lr_step_size 30 --decay 1e-5 \
                                                  --batch_size 64 --num_workers 4 --data_size ${datasize} \
                                                  --dataset ${dataset} --label_split_dir splits \
                                                  --label_type ${label_tp} --output_pred \
                                                  --model ${model} \
                                                  --epochs ${epoch} \
                                                  --sensor_type ${sensor_tp} --eval
            done
          done
        done
      done
    done
  done
done