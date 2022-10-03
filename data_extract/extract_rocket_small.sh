#!/bin/bash

echo "Bag file: $1"

python3 ./bag_extractor.py --bag-file=/gpfs/space/projects/Bolt/bagfiles/$1 --extract-dir=$2 --crop-camera-images --resize-camera-images --extract-side-cameras --extract-lidar