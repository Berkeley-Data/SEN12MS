#!/bin/bash

for lr in 0.001 0.0005 0.0001 0.00005 0.00001
do
  for epoch in 200
  do
    for label_tp in single_label multi_label
    do
      for model in Moco # Moco_1x1RND Moco_1x1
      do
        python classification/main_train.py --exp_name finetune --simple_scheme \
                                            --lr ${lr} --use_lr_step --lr_step_size 30 --decay 1e-5 \
                                            --pt_name electric-mountain-33 --pt_dir pretrained/moco \
                                            --batch_size 64 --num_workers 4 --data_size 1024 \
                                            --dataset sen12ms --label_split_dir splits \
                                            --label_type ${label_tp} \
                                            --model ${model} \
                                            --epochs ${epoch} \
                                            --sensor_type s1s2 --eval
      done
    done
  done
done
#