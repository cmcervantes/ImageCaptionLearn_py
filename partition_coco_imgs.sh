#!/usr/bin/env bash

img_id_file="coco_sub_train_imgs.txt"
while IFS= read -r img_id
do
    cp "/home/ccervan2/source/data/MSCOCO/train2014/$img_id" "/home/ccervan2/source/data/MSCOCO/train2014_sub/"
done < "$img_id_file"