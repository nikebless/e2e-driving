import argparse
import math
import os
import shutil
import sys
from pathlib import Path
import logging
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch
from moviepy.editor import ImageSequenceClip
from skimage import io
from torchvision import transforms
from tqdm.auto import tqdm

from dataloading.nvidia import NvidiaDataset, Normalize
from pilotnet import PilotNetConditional, PilotnetControl
import trainer as trainers
import train
from velocity_model.velocity_model import VelocityModel
import onnxruntime as ort


# red, blue, yellow
PRED_COLORS = [(255, 0, 0), (0, 0, 255), (255, 255, 0)]
PRED_SIZES = [9, 7, 5]


def create_driving_video(dataset_folder):
    dataset_path = Path(dataset_folder)
    dataset = NvidiaDataset([dataset_path], metadata_file="nvidia_frames.csv")

    temp_frames_folder = dataset_path / 'temp'
    shutil.rmtree(temp_frames_folder, ignore_errors=True)
    temp_frames_folder.mkdir()

    draw_driving_frames(dataset, temp_frames_folder)
    output_video_path = dataset_path / 'video.mp4'
    convert_frames_to_video(temp_frames_folder, output_video_path, fps=30)

    shutil.rmtree(temp_frames_folder, ignore_errors=True)

    print(f"{dataset.name}: output video {output_video_path} created.")


def create_prediction_video(dataset_folder, model_paths, model_types, model_names, config: train.TrainingConfig):
    dataset_path = Path(dataset_folder)
    save_path = Path('./')
    dataset = NvidiaDataset([Path('/data/Bolt/dataset/2021-10-14-13-08-51_e2e_rec_vahi_backwards/')], name=dataset_path.name)

    #dataset.frames = dataset.frames[9160:23070]

    temp_frames_folder = save_path / 'temp'
    shutil.rmtree(temp_frames_folder, ignore_errors=True)
    temp_frames_folder.mkdir()

    steering_predictions = {}
    trajectories = {}

    for model_path, model_type, model_name in zip(model_paths, model_types, model_names):

        model_steering_predictions = get_steering_predictions(dataset_path, model_path, model_type, config)
        print(f'model: {model_path}. Steering predictions:', model_steering_predictions.shape, type(model_steering_predictions))
        speed_predictions = get_speed_predictions(dataset)
        steering_predictions[model_name] = model_steering_predictions

    draw_prediction_frames(dataset, steering_predictions, speed_predictions, temp_frames_folder)

    output_video_path = save_path / f"{str(Path(model_path).parent.name)}.mp4"
    convert_frames_to_video(temp_frames_folder, output_video_path, fps=30)

    shutil.rmtree(temp_frames_folder, ignore_errors=True)

    print(f"{dataset.name}: output video {output_video_path} created.")


def get_steering_predictions(dataset_path, model_path, model_type, config):
    print(f"{dataset_path.name}: steering predictions")

    tr = transforms.Compose([Normalize()])
    # TODO: remove hardcoded path
    dataset = NvidiaDataset([Path(dataset_path)],
                            tr, name=dataset_path.name)
    validloader_tr = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False,
                                         num_workers=16, pin_memory=True, persistent_workers=True)

    steering_predictions = []

    if Path(model_path).suffix == '.onnx':
        model = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        input_name = model.get_inputs()[0].name
        steering_predictions = []

        for batch_idx, (data, _, __) in enumerate(validloader_tr):
            inputs = data['image'].numpy()
            outs = model.run(None, {input_name: inputs })[0]
            steering_predictions.append(outs.squeeze())

            if batch_idx > 10:
                break

        steering_predictions = np.array(steering_predictions).flatten()
    else:
        #trainer.force_cpu()  # not enough memory on GPU for parallel processing  # TODO: make input argument
        if model_type == "pilotnet-ebm":
            trainer = trainers.EBMTrainer(train_conf=config)
            model = trainer.model
        else:
            print(f"Unknown model type '{model_type}'")
            sys.exit()

        model.load_state_dict(torch.load(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        steering_predictions = trainer.predict(model, validloader_tr)

    return steering_predictions


def get_speed_predictions(dataset):
    print(f"{dataset.name}: speed predictions")
    velocity_model = VelocityModel(positions_parquet='velocity_model/summer2021-positions.parquet')

    frames = dataset.frames

    x = frames["position_x"]# + np.random.normal(0, 0.1, len(frames))
    y = frames["position_y"]# + np.random.normal(0, 0.1, len(frames))
    yaw = frames["yaw"]# + np.random.normal(0, 0.2, len(frames))

    result_df = pd.DataFrame(data={'x': x, 'y': y, 'yaw': yaw})
    result_df = result_df.fillna(0)  # TODO: correct NaN handling
    speed_predictions = result_df.apply(
        lambda df: velocity_model.find_speed_for_position(df['x'], df['y'], df['yaw'])[0], axis=1)
    return speed_predictions.to_numpy()


def draw_steering_angle(frame, steering_angle, steering_wheel_radius, steering_position, size, color):
    steering_angle_rad = math.radians(steering_angle)
    x = steering_wheel_radius * np.cos(np.pi / 2 + steering_angle_rad)
    y = steering_wheel_radius * np.sin(np.pi / 2 + steering_angle_rad)
    return cv2.circle(frame, (steering_position[0] + int(x), steering_position[1] - int(y)), size, color, thickness=-1)


def draw_prediction_frames_wp(dataset, trajectory, temp_frames_folder):
    print("Creating video frames.")

    dataset.frames["turn_signal"].fillna(99, inplace=True)  # TODO correct NaN handling

    t = tqdm(enumerate(dataset), total=len(dataset))
    t.set_description(dataset.name)

    for frame_index, (data, target_values, condition_mask) in t:
        frame = data["image"].permute(1, 2, 0).cpu().numpy()
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        true_angle = math.degrees(data["steering_angle"])
        true_speed = data["vehicle_speed"] * 3.6

        true_waypoints = data["waypoints"]
        predicted_waypoints = trajectory[frame_index]

        position_x = data["position_x"]
        position_y = data["position_y"]
        yaw = math.degrees(data["yaw"])
        turn_signal = int(data["turn_signal"])

        cv2.putText(frame, 'True: {:.2f} deg, {:.2f} km/h'.format(true_angle, true_speed), (10, 1150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

        cv2.putText(frame, 'frame: {}'.format(frame_index), (10, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, 'x: {:.2f}'.format(position_x), (10, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, 'y: {:.2f}'.format(position_y), (10, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, 'yaw: {:.2f}'.format(yaw), (10, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

        turn_signal_map = {
            1: "straight",
            2: "left",
            0: "right"
        }
        cv2.putText(frame, 'turn signal: {}'.format(turn_signal_map.get(turn_signal, "unknown")), (10, 1100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

        draw_trajectory(frame, true_waypoints, (0, 255, 0))
        draw_trajectory(frame, predicted_waypoints, (255, 0, 0))

        io.imsave(f"{temp_frames_folder}/{frame_index + 1:05}.jpg", frame)


def draw_prediction_frames(dataset, predicted_angles, predicted_speed, temp_frames_folder):
    print("Creating video frames.")

    #dataset.frames = dataset.frames[9160:23070]
    #dataset.frames = dataset.frames[9160:10070]

    dataset.frames["turn_signal"].fillna(99, inplace=True)  # TODO correct NaN handling

    t = tqdm(enumerate(dataset), total=len(dataset))
    t.set_description(dataset.name)

    for frame_index, (data, target_values, condition_mask) in t:
        frame = data["image"].permute(1, 2, 0).cpu().numpy()
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        true_angle = math.degrees(data["steering_angle"])
        pred_angles = { model_name: math.degrees(preds[frame_index]) for model_name, preds in predicted_angles.items() }
        true_speed = data["vehicle_speed"] * 3.6
        pred_speed = predicted_speed[frame_index] * 3.6

        position_x = data["position_x"]
        position_y = data["position_y"]
        yaw = math.degrees(data["yaw"])
        turn_signal = int(data["turn_signal"])

        turn_signal_map = {
            1: "straight",
            2: "left",
            0: "right"
        }
        

        start_text_top_y = 850

        cv2.putText(frame, 'frame: {}'.format(frame_index), (10, start_text_top_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, 'x: {:.2f}'.format(position_x), (10, start_text_top_y+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, 'y: {:.2f}'.format(position_y), (10, start_text_top_y+50*2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, 'yaw: {:.2f}'.format(yaw), (10, start_text_top_y+50*3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, 'turn signal: {}'.format(turn_signal_map.get(turn_signal, "unknown")), (10, start_text_top_y+50*4),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, 'True: {:.2f} deg, {:.2f} km/h'.format(true_angle, true_speed), (10, start_text_top_y+50*5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        
        for i, (model_name, pred_angle) in enumerate(pred_angles.items()):
            cv2.putText(frame, 'Pred: {:.2f} deg ({}), {:.2f} km/h'.format(pred_angle, model_name, pred_speed), (10, start_text_top_y+50*(6+i)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, PRED_COLORS[i], 2,
                        cv2.LINE_AA)

        
        

        radius = 200
        steering_pos = (960, 1200)
        cv2.circle(frame, steering_pos, radius, (255, 255, 255), 7)

        cv2.rectangle(frame, (905, 1200), (955, 1200 - int(3 * true_speed)), (0, 255, 0), cv2.FILLED)
        cv2.rectangle(frame, (965, 1200), (1015, 1200 - int(3 * pred_speed)), (255, 0, 0), cv2.FILLED)

        draw_steering_angle(frame, true_angle, radius, steering_pos, 13, (0, 255, 0))

        for i, (_, pred_angle) in enumerate(pred_angles.items()):
            draw_steering_angle(frame, pred_angle, radius, steering_pos, PRED_SIZES[i], PRED_COLORS[i])

        #frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        io.imsave(f"{temp_frames_folder}/{frame_index + 1:05}.jpg", frame)


def draw_driving_frames(dataset, temp_frames_folder):
    print(f"Drawing driving frames ")
    t = tqdm(enumerate(dataset), total=len(dataset))
    t.set_description(dataset.name)
    for frame_index, (data, target_values, condition_mask) in t:

        frame = data["image"].permute(1, 2, 0).cpu().numpy()
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        true_angle = math.degrees(data["steering_angle"])
        true_speed = data["vehicle_speed"] * 3.6
        autonomous = data["autonomous"]

        if autonomous:
            color = (255, 0, 0)
            cv2.putText(frame, 'Mode:    AUTONOMOUS', (10, 1100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        else:
            color = (0, 255, 0)
            cv2.putText(frame, 'Mode:    MANUAL'.format(true_angle), (10, 1100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2,
                        cv2.LINE_AA)

        cv2.putText(frame, 'Steering: {:.2f} deg'.format(true_angle), (10, 1150), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2,
                    cv2.LINE_AA)
        cv2.putText(frame, 'Speed:   {:.2f} km/h'.format(true_speed), (10, 1200), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2,
                    cv2.LINE_AA)

        radius = 200
        steering_pos = (960, 1200)
        cv2.circle(frame, steering_pos, radius, (255, 255, 255), 7)

        draw_steering_angle(frame, true_angle, radius, steering_pos, 13, color)
        cv2.rectangle(frame, (935, 1200), (980, 1200 - int(3 * true_speed)), color, cv2.FILLED)

        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        io.imsave(f"{temp_frames_folder}/{frame_index + 1:05}.jpg", frame)


def draw_trajectory(frame, waypoints, color):
    scale = 5
    for (x, y) in zip(waypoints[0::2], waypoints[1::2]):
        cv2.circle(frame, (935 - int(scale * y), 1200 - int(scale * x)), 3, color, 5)


def convert_frames_to_video(frames_folder, output_video_path, fps=30):
    output_folder = Path(os.path.split(output_video_path)[:-1][0])
    output_folder.mkdir(parents=True, exist_ok=True)

    p = Path(frames_folder).glob('**/*.jpg')
    image_list = sorted([str(x) for x in p if x.is_file()])

    print("Creating video {}, FPS={}".format(frames_folder, fps))
    clip = ImageSequenceClip(image_list, fps=fps)
    clip.write_videofile(str(output_video_path))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '--dataset-folder',
        required=True,
        help='Path to a dataset extracted from a bag file'
    )

    argparser.add_argument(
        '--video-type',
        required=True,
        choices=['driving', 'prediction'],
        help="Type of the video, 'driving' or 'prediction'."
    )
    argparser.add_argument(
        '--model-path',
        nargs='+', # 1 or more
        help="Path to pytorch model to use for creating steering predictions."
    )

    argparser.add_argument(
        '--model-type',
        required=False,
        nargs='+', # 1 or more
        default=['pilotnet'],
        choices=['pilotnet', 'pilotnet-conditional', 'pilotnet-control', 'pilotnet-ebm'],
    )

    argparser.add_argument(
        '--model-name',
        required=False,
        nargs='+',
        default=['pilotnet'],
        help='Name of the model used for saving model and logging in W&B.'
    )

    argparser.add_argument(
        '--camera-name',
        required=False,
        default="front_wide",
        choices=['front_wide', 'left', 'right', 'all'],
        help="Camera to use for training. Only applies to 'nvidia-camera' modality."
    )

    argparser.add_argument(
        '--num-waypoints',
        type=int,
        default=10,
        help="Number of waypoints used for trajectory."
    )

    argparser.add_argument(
        '--batch-size',
        type=int,
        default=512,
        help='Weight decay used in training.'
    )

    argparser.add_argument(
        '--num-workers',
        type=int,
        default=16,
        help='Number of workers used for data loading.'
    )

    argparser.add_argument(
        '--ebm-inference-samples',
        type=int,
        default=128, 
        help='Number of samples used for test-time EBM inference.'
    )

    argparser.add_argument(
        '--debug',
        action='store_true',
        help='When true, debug mode is enabled.'
    )

    args = argparser.parse_args()
    dataset_folder = args.dataset_folder
    video_type = args.video_type
    args.learning_rate = 0
    args.weight_decay = 0
    args.patience = 0
    args.max_epochs = 0
    args.batch_sampler = None
    args.wandb_project = ''
    args.loss = 'ebm'
    args.ebm_train_samples = 0
    args.pretrained_model = False

    conf = train.TrainingConfig(args)
    print("Creating video from: ", dataset_folder)

    if video_type == 'driving':
        create_driving_video(dataset_folder)
    elif video_type == 'prediction':
        create_prediction_video(dataset_folder, model_paths=args.model_path, model_types=args.model_type, model_names=args.model_name, config=conf)
