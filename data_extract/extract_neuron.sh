#!/bin/bash

echo "Bag file: $1"

python3 ./bag_extractor.py --bag-file=/data/Bolt/bagfiles/$1 --extract-dir=$2