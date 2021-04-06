#!/bin/bash

for lr in 0.00001 0.00005 0.0001 0.001 0.001
do
  for epoch in 200
  do
    for label_tp in single_label multi_label
    do
      for model in Supervised_1x1 Supervised
      do
        python classification/main_train.py --exp_name sup_learning --IGBP_simple \
                                            --lr ${lr} --use_lr_step --lr_step_size 30 --decay 1e-5 \
                                            --batch_size 64 --num_workers 4 --data_size 1024 \
                                            --data_dir data/sen12ms/data --label_split_dir splits \
                                            --label_type ${label_tp} \
                                            --model ${model} \
                                            --epochs ${epoch} \
                                            --use_s1 --use_s2 --eval
      done
    done
  done
done
#