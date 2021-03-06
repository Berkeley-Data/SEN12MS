#!/bin/bash

export WANDB_PROJECT=sup_scene_cls

for datasize in 1024 # 512 256
do
  for sensor_tp in s1 s2 # s1s2
  do
    for dataset in bigearthnet # sen12ms
    do
      for lr in 0.001 0.0005 0.0001 0.00005 0.00001
      do
        for epoch in 300
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