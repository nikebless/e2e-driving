#!/bin/bash

echo "Bag file: $1"

python ./bag_extractor.py --bag-file=/gpfs/space/projects/Bolt/bagfiles/$1 --extract-dir=$2