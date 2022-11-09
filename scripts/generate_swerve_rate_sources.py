import os, sys
import dotenv
dotenv.load_dotenv('../.env')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator, FixedLocator
import matplotlib.patheffects as pe

from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

sys.path.append('../')
from common import BOLT_DIR
from dataloading.nvidia import NvidiaDataset


WANDB_ENTITY = os.getenv('WANDB_ENTITY')
WANDB_PROJECT = os.getenv('WANDB_PROJECT')



def are_locations_close(loc_a, loc_b, threshold=50):
    return np.linalg.norm(loc_a - loc_b) < threshold

def get_closest_frame_by_loc(df, target_loc):
    locations = df[['position_x', 'position_y']].to_numpy().astype(np.float32)
    df['distance_to_target'] = np.linalg.norm(locations - target_loc)
    return df.loc[df['distance_to_target'].idxmin()]

def get_closest_row_idx_by_timestamp(df, dt):
    df['timestamp'] = pd.to_datetime(df['index'])
    return (abs(df['timestamp'] - dt)).idxmin()

def get_longest_intervention_periods(df):
    df['autonomous_next'] = df['autonomous'].shift(-1)
    starts_ends_df = df[(df['autonomous'] & (df['autonomous_next'] == False)) | ((df['autonomous'] == False) & df['autonomous_next'])]
    starts_ends = [row['row_id'] for i, row in starts_ends_df.iterrows()]
    starts = np.array(starts_ends)[::2]
    ends = np.array(starts_ends)[1::2]
    longest_idxs = np.argsort(ends - starts)
    return (starts[longest_idxs], ends[longest_idxs])

def split_back_forth_drive_into_two(dataset):

    frames_df = dataset.frames
    vehicle_cmd_df = dataset.vehicle_cmd_frames
    # find the longest intervention period
    found_direction_change = False
    for forward_end, forward_start in zip(*get_longest_intervention_periods(frames_df)):
        if are_locations_close(frames_df[frames_df['row_id'] == forward_end][['position_x', 'position_y']].to_numpy(), track_direction_change_location) or \
            are_locations_close(frames_df[frames_df['row_id'] == forward_start][['position_x', 'position_y']].to_numpy(), track_direction_change_location):
            found_direction_change = True
            break

    if not found_direction_change:
        print('Couldn\'t find the longest intervention in the track direction change location')
        return None

    # split the drive into two
    df1 = frames_df[frames_df['row_id'] <= forward_end]
    df2 = frames_df[frames_df['row_id'] > forward_start]

    forward_end_ts = pd.to_datetime(df1.iloc[-1]['index'])
    backward_start_ts = pd.to_datetime(df2.iloc[0]['index'])

    forward_end_idx = get_closest_row_idx_by_timestamp(vehicle_cmd_df, forward_end_ts)
    backward_end_idx = get_closest_row_idx_by_timestamp(vehicle_cmd_df, backward_start_ts)

    df1_vehicle_cmd = vehicle_cmd_df.iloc[:forward_end_idx]
    df2_vehicle_cmd = vehicle_cmd_df.iloc[backward_end_idx:]

    # save the pandas dataframes back into NvidiaDataset objects
    dataset_forward = deepcopy(dataset)
    dataset_backward = deepcopy(dataset)
    dataset_forward.frames = df1
    dataset_forward.vehicle_cmd_frames = df1_vehicle_cmd
    dataset_backward.frames = df2
    dataset_backward.vehicle_cmd_frames = df2_vehicle_cmd

    return dataset_forward, dataset_backward

def steering_angle_to_bin_idx(angle_degs, num_samples=512, bound=4.5):
    angle_rads = np.radians(angle_degs)
    bin_idx = (angle_rads) * (num_samples-1) / (bound*2) + (num_samples-1)/2
    return bin_idx

def bin_idx_to_steering_angle(bin_idx, num_samples=512, bound=4.5):
    angle_rads = (bin_idx - (num_samples-1)/2) * bound*2 / (num_samples-1)
    angle_degs = np.degrees(angle_rads)
    return angle_degs

from copy import deepcopy

DURATION_SEC = 5

def crop_place(ds_forward, ds_backward, place='village-forward'):
    dataset = ds_forward if 'forward' in place else ds_backward
    intersection_ds = deepcopy(dataset)
    inter_fr = dataset.frames

    if place == 'village-forward':
        intersection_start = inter_fr[inter_fr['position_x'] < -8040].iloc[0]
    elif place == 'village-backward':
        intersection_start = inter_fr[inter_fr['position_x'] > -8085].iloc[0]
    elif place == 'early-forward':
        intersection_start = inter_fr[inter_fr['position_y'] > -3495].iloc[0]
    elif place == 'early-backward':
        intersection_start = inter_fr[inter_fr['position_x'] > -9075].iloc[0]
    elif place == 't-intersection-forward':
        intersection_start = inter_fr[inter_fr['position_y'] > -2360].iloc[0]
    elif place == 't-intersection-backward':
        intersection_start = inter_fr[inter_fr['position_x'] > -7665].iloc[0]
    elif place == 'sharp-turn-forward':
        intersection_start = inter_fr[inter_fr['position_y'] > -1800].iloc[0]

    inter_fr = inter_fr[inter_fr['row_id'] >= intersection_start['row_id']]
    inter_fr = inter_fr[inter_fr['row_id'] <= intersection_start['row_id'] + DURATION_SEC * 30]
    intersection_ds.frames = inter_fr
    return intersection_ds

DURATION_SEC = 5
STEERING_RATIO = 14.7
num_samples = 512
bound = 4.5

def make_intersection_plot(ds, name):

    fig, ax = plt.subplots(figsize=(10, 10))

    eff_angles = ds.frames.steering_angle
    pred_angles = ds.frames.cmd_steering_angle * STEERING_RATIO

    ax.set_xlim(0, num_samples)
    ax.set_ylim(DURATION_SEC*30+1, 0) # 211 = 7 seconds * 30 fps + 1
    ax.set_aspect(2.6)

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel('Steering Angle')
    ax.set_ylabel('Time (s)')
    tick_formatter = FuncFormatter(lambda x, pos: int(bin_idx_to_steering_angle(x, num_samples=num_samples, bound=bound).round()))
    tick_locator = FixedLocator(steering_angle_to_bin_idx([-225, -180, -135, -90, -45, 0, 45, 90, 135, 180, 225]))
    ax.xaxis.set_major_formatter(tick_formatter)
    ax.xaxis.set_major_locator(tick_locator)
    ax.yaxis.set_major_locator(MultipleLocator(30))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(x)/30))

    true_angles_indices = steering_angle_to_bin_idx(np.degrees(eff_angles), num_samples=num_samples, bound=bound)
    pred_angles_indices = steering_angle_to_bin_idx(np.degrees(pred_angles), num_samples=num_samples, bound=bound)

    ax.plot(true_angles_indices, np.arange(0, len(true_angles_indices)), color='white', linestyle='dashed', linewidth=2, label='Effective angle', path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
    ax.plot(pred_angles_indices, np.arange(0, len(pred_angles_indices)), color='red', linestyle='dashed', linewidth=2, label='Model prediction')
    ax.set_title('stuf')
    ax.legend()
    fig.savefig(name)


track_direction_change_location = np.array([-9683.68050786, -1542.68155186])
root_path = Path(BOLT_DIR) / 'end-to-end/drives-ebm-paper'
expert_ds = NvidiaDataset([root_path / '2021-10-26-10-49-06_e2e_rec_ss20_elva_eval_chunk'])
expert_back_ds = NvidiaDataset([root_path / '2021-10-26-11-08-59_e2e_rec_ss20_elva_back_eval_chunk'])



df = pd.read_csv('ebm-experiments-final-results.csv')
# convert str to tuple
df['drive'] = df['drive'].apply(lambda x: tuple(x[1:-1].replace('\'', '').split(',')))
df['drive'] = df['drive'].apply(lambda x: [i.strip() for i in x if i != ''])


datasets_forward = {}
datasets_backward = {}

for i, row in tqdm(df.iterrows()):
    drives = row['drive']
    ds_forward = None
    ds_backward = None
    forward_metrics = {}
    backward_metrics = {}
    title_forward = None
    title_backward = None
    print(type(drives), drives)

    if len(drives) == 1:
        ds_combined = NvidiaDataset([root_path / drives[0]])
        if drives[0] not in ['2022-09-09-11-47-04', '2022-09-09-10-51-33-mdn-1-s1']: # interrupted single direction drives
            ds_forward, ds_backward = split_back_forth_drive_into_two(ds_combined)
            title_forward = drives[0] + ' (forward)'
            title_backward = drives[0] + ' (backward)'
        else:
            ds_forward = ds_combined
            title_forward = drives[0]
    elif len(drives) == 2:
        ds_forward = NvidiaDataset([root_path / drives[0]])
        ds_backward = NvidiaDataset([root_path / drives[1]])
        title_forward = drives[0]
        title_backward = drives[1]
    elif len(drives) == 3:
        assert '2022-08-31-15-18-55_elva_classifier_512_forward_continued' in drives[1]
        ds_forward = NvidiaDataset([root_path / drives[0], root_path / drives[1]])
        ds_backward = NvidiaDataset([root_path / drives[2]])
        title_forward = drives[0] + ' + continued'
        title_backward = drives[2]

    if ds_forward is not None and title_forward is not None:
        datasets_forward[title_forward] = ds_forward
    if ds_backward is not None and title_backward is not None:
        datasets_backward[title_backward] = ds_backward

MODEL_NAME_LIST = ['mae-s2', 'ebm-512-s1', 'ebm-normal-1-s1', 'ebm-spatial-0-s2', 'mdn-5-s1', 'classifier-512']

def model_from_drive_name(drive_name):
    for model_name in MODEL_NAME_LIST:
        if model_name in drive_name:
            return model_name
    return None

def count_interventions(frames):
    frames['autonomous_next'] = frames.shift(-1)['autonomous']
    return len(frames[frames.autonomous & (frames.autonomous_next == False)])
    

PLACES = ['village-forward', 'early-forward', 'early-backward'] # 'village-backward', 't-intersection-forward', 't-intersection-backward']

os.makedirs('plots', exist_ok=True)

progress = tqdm(total=len(PLACES)*len(datasets_forward))

for (forward_name, forward_ds), (backward_name, backward_ds) in zip(datasets_forward.items(), datasets_backward.items()):
    model_name = model_from_drive_name(forward_name)

    os.makedirs(f'plots/{model_name}', exist_ok=True)

    for place in PLACES:
        place_ds = crop_place(forward_ds, backward_ds, place)
        # has_intervention = count_interventions(place_ds.frames) > 0
        # print(f'{place}_{forward_name} —— has intervention: {has_intervention}')
        make_intersection_plot(place_ds, f'plots/{model_name}/{place}_{forward_name}.png')
        progress.update(1)

# zip the plots folder

