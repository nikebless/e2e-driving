#!/bin/bash

echo "Camera v1"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality nvidia-camera --drive-datasets 2021-11-25-12-09-43_e2e_rec_elva-nvidia-v1-0.8 2021-11-25-12-21-17_e2e_rec_elva-nvidia-v1-0.8-forward
echo "Camera v2"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality nvidia-camera --drive-datasets 2021-11-25-14-01-46_e2e_rec_elva-licamera-v2-0.8-back 2021-11-25-14-13-59_e2e_rec_elva-licamera-v2-0.8-forward
echo "Camera v3"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality nvidia-camera --drive-datasets 2021-11-25-14-51-46_e2e_rec_elva-l-camera-v3-0.8-back 2021-11-25-15-04-26_e2e_rec_elva-l-camera-v3-0.8-forward
echo "Camera in train"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality nvidia-camera --drive-datasets 2021-11-25-13-11-40_e2e_rec_elva-licamera-inTrain-0.8-back 2021-11-25-13-24-00_e2e_rec_elva-licamera-inTrain-0.8-forward

echo "Lidar v1"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2021-11-25-12-45-35_e2e_rec_elva-lidar-v1-0.8-back 2021-11-25-12-57-24_e2e_rec_elva-lidar-v1-0.8-forward
echo "Lidar v2"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2021-11-25-14-27-56_e2e_rec_elva-lidar-v2-0.8-back 2021-11-25-14-39-43_e2e_rec_elva-lidar-v2-0.8-forward
echo "Lidar v3"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2021-11-25-15-16-31_e2e_rec_elva-l-lidar-v3-0.8-back 2021-11-25-15-27-38_e2e_rec_elva-l-lidar-v3-0.8-forward
echo "Lidar in train"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2021-11-25-13-37-42_e2e_rec_elva-lilidar-inTrain-0.8-back 2021-11-25-13-48-44_e2e_rec_elva-lilidar-inTrain-0.8-forward
echo "Lidar in train night"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2021-11-25-16-57-26_e2e_rec_elva-lidar-inTrain-0.8-forwardNight 2021-11-25-17-08-28_e2e_rec_elva-lidar-inTrain-0.8-backNight
echo "Lidar night"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2021-11-25-17-31-42_e2e_rec_elva-lidar-0.8-forwardNight 2021-11-25-17-43-47_e2e_rec_elva-lidar-0.8-backNight
echo "Lidar night #2"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2021-11-25-17-56-16_e2e_rec_elva-lidar-0.8-forwardNight_attempt2 2021-11-25-18-07-28_e2e_rec_elva-lidar-0.8-backNight_attempt2

echo "Lidar all channels"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2021-11-26-11-19-15_e2e_rec_elva_i_allChannels_forward_0.8 2021-11-26-11-30-23_e2e_rec_elva_i_allChannels_back_0.8
echo "Lidar intensity"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2021-11-26-10-53-35_e2e_rec_elva_intensity_forward_0.8 2021-11-26-11-07-10_e2e_rec_elva_intensity_back_0.8
echo "Lidar range"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2021-11-26-11-42-02_e2e_rec_elva_i_range_forward_0.8
echo "Lidar ambience"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2021-11-26-11-53-18_e2e_rec_elva_i_ambience_forward_0.8

echo "Lidar Winter v1"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2022-02-02-10-39-23_e2e_rec_elva_winter_lidar_forward_08 2022-02-02-10-50-07_e2e_rec_elva_winter_lidar_forward_08 2022-02-02-10-53-16_e2e_rec_elva_winter_lidar_backw_08
echo "Lidar Winter v2"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2022-02-02-11-05-18_e2e_rec_elva_winter_lidar-v5_forw_08 2022-02-02-11-18-14_e2e_rec_elva_winter_lidar-v5_backw_08
echo "Lidar Winter v3"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality ouster-lidar --drive-datasets 2022-02-02-11-32-37_e2e_rec_elva_winter_lidar-v3_forw_08 2022-02-02-11-45-34_e2e_rec_elva_winter_lidar-v3_backw_08
echo "Camera Winter v3"
python metrics.py --root-path /gpfs/space/projects/Bolt/dataset-paper --input-modality nvidia-camera --drive-datasets 2022-02-02-11-58-48_e2e_rec_elva_winter_camera-v3_forw_08


