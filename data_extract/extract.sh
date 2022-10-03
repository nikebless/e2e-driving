#!/bin/bash

echo "Bag file: $1"

python3 ./bag_extractor.py --bag-file=$1 --extract-dir=$2 --extract-side-cameras