#!/bin/bash

export WANDB_PROJECT=scene_classification

for dataset in sen12ms # bigearthnet
do
  for lr in 0.001 0.0005 0.0001 0.00005 0.00001
  do
    for epoch in 300
    do
      for label_tp in single_label multi_label
      do
        for model in Moco_1x1 Moco_1x1RND # Moco
        do
          python classification/main_train.py --simple_scheme \
                                              --lr ${lr} --use_lr_step --lr_step_size 30 --decay 1e-5 \
                                              --batch_size 64 --num_workers 4 --data_size 1024 \
                                              --dataset ${dataset} --label_split_dir splits \
                                              --label_type ${label_tp} \
                                              --model ${model} --pt_dir pretrained/moco \
                                              --epochs ${epoch} --eval \
                                              --exp_name finetune --sensor_type s2 \
                                              --pt_name sen12_x_aug_ep1000
        done
      done
    done
  done
done