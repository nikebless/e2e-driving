from collections import defaultdict
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['EGL_DEVICE_ID'] = os.environ.get('CUDA_AVAILABLE_DEVICES', '0')

import time
import numpy as np
import math
import argparse
import wandb
import traceback
import dotenv
dotenv.load_dotenv()

import vista
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from vista.entities.agents.Dynamics import steering2curvature

from common import BOLT_DIR, LEXUS_LENGTH, LEXUS_WIDTH, LEXUS_WHEEL_BASE, LEXUS_STEERING_RATIO, \
                   FULL_IMAGE_HEIGHT, FULL_IMAGE_WIDTH, \
                   IMAGE_CROP_XMIN, IMAGE_CROP_XMAX, IMAGE_CROP_YMIN, IMAGE_CROP_YMAX, \
                   OnnxSteeringModel, Timing
from vista_sim.video import VideoStream
from vista_sim.dynamics_model import OnnxDynamicsModel


LOG_FREQUENCY_SEC = 1
FPS = 10
WANDB_ENTITY = os.environ.get('WANDB_ENTITY', None)
WANDB_PROJECT = os.environ.get('WANDB_PROJECT', None)
TRACES_ROOT = os.path.join(BOLT_DIR, 'end-to-end', 'vista')
FRAME_START_OFFSET = 100
ROAD_WIDTH = 3.5 # TODO: change to 2.5
MAX_OPENGL_RETRIES = int(os.environ.get('MAX_OPENGL_RETRIES', 10))
VISTA_DOWNSAMPLE_FACTOR = 4


def step_sensors_safe(car, timings):
    successful = False
    try: 
        car.step_sensors(timings)
        successful = True
    except Exception as err:
        print(err)
        traceback.print_exc()
    return successful


def vista_step(car, curvature=None, speed=None, timings=dict()):
    if curvature is None: 
        curvature = car.trace.f_curvature(car.timestamp)
    if speed is None: 
        speed = car.trace.f_speed(car.timestamp)
    
    with Timing(timings, 'step_dynamics'):
        car.step_dynamics(action=np.array([curvature, speed]), dt=1/FPS, timings=timings)

    with Timing(timings, 'step_sensors'):
        n_attempts = 0
        while not step_sensors_safe(car, timings) and n_attempts < MAX_OPENGL_RETRIES:
            n_attempts += 1
            print(f'Waiting 5 sec before another step_sensors() attempt... (tried {n_attempts}/{MAX_OPENGL_RETRIES} times)')
            time.sleep(5)

        if n_attempts == MAX_OPENGL_RETRIES:
            print('Giving up after too many OpenGL errors')
            car.done = True


def check_out_of_lane(car):
    distance_from_center = np.abs(car.relative_state.x)
    road_width = car.trace.road_width 
    half_road_width = road_width / 2
    return distance_from_center > half_road_width

# def resize(img):
#     scale = 0.2
#     height = IMAGE_CROP_YMAX - IMAGE_CROP_YMIN
#     width = IMAGE_CROP_XMAX - IMAGE_CROP_XMIN

#     scaled_width = int(width * scale)
#     scaled_height = int(height * scale)

#     return cv2.resize(img, dsize=(scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)

def resize(cv_img, antialias):
    scale = 0.2
    height = IMAGE_CROP_YMAX - IMAGE_CROP_YMIN
    width = IMAGE_CROP_XMAX - IMAGE_CROP_XMIN

    scaled_width = int(width * scale)
    scaled_height = int(height * scale)

    img = torch.tensor(cv_img, dtype=torch.uint8).permute(2, 0, 1)
    img = F.resize(img, (scaled_height, scaled_width), antialias=antialias, interpolation=InterpolationMode.BILINEAR)
    return img.permute(1, 2, 0).numpy()

def normalise(img):
    return (img / 255.0)

# def crop(img):
#     return img[IMAGE_CROP_YMIN:IMAGE_CROP_YMAX, IMAGE_CROP_XMIN:IMAGE_CROP_XMAX, :]


def crop(cv_img):
    crop_xmin = IMAGE_CROP_XMIN // VISTA_DOWNSAMPLE_FACTOR
    crop_xmax = IMAGE_CROP_XMAX // VISTA_DOWNSAMPLE_FACTOR
    crop_ymin = IMAGE_CROP_YMIN // VISTA_DOWNSAMPLE_FACTOR
    crop_ymax = IMAGE_CROP_YMAX // VISTA_DOWNSAMPLE_FACTOR
    return cv_img[crop_ymin:crop_ymax, crop_xmin:crop_xmax, :]

def crop_after_resize(cv_img, scale=0.2):
    crop_xmin = int(IMAGE_CROP_XMIN * scale)
    crop_xmax = int(IMAGE_CROP_XMAX * scale)
    crop_ymin = int(IMAGE_CROP_YMIN * scale)
    crop_ymax = int(IMAGE_CROP_YMAX * scale)

    return cv_img[crop_ymin:crop_ymax, crop_xmin:crop_xmax, :]

def preprocess(full_obs, antialias, resize_mode):

    if resize_mode == 'full' or resize_mode == 'downsample':
        img = crop( full_obs )
        img = resize( img, antialias=antialias)
    elif resize_mode == 'resize':
        # full_obs already resized, only crop
        img = crop_after_resize( full_obs )
    elif resize_mode == 'resize_and_crop':
        # full_obs already resized and cropped
        img = full_obs

    img = normalise( img )
    return img.astype(np.float32)

def grab_and_preprocess_obs(car, camera, antialias, resize_mode):
    full_obs = car.observations[camera.name]
    obs = preprocess(full_obs, antialias, resize_mode)
    return obs

def run_episode(model, world, camera, car, antialias, save_video=False, video_name='model_run.avi', resize_mode='full', dynamics_model=None):
    stream = VideoStream(FPS, suffix='_full') # TODO: change tmp folder generation to be more robust
    stream_cropped = VideoStream(FPS, suffix='_cropped', no_encoding=True)
    i_step = 0

    world.reset()
    car.reset(0, 0, FRAME_START_OFFSET)
    display.reset()
    observation = grab_and_preprocess_obs(car, camera, antialias, resize_mode)


    last_driven_frame_idx = 0
    crash_times = []

    timings = dict()

    while True:

        inference_start = time.perf_counter()
        model_input = np.moveaxis(observation, -1, 0)
        model_input = np.expand_dims(model_input, axis=0)
        predictions = model.predict(model_input)
        steering_angle = predictions.item()
        if dynamics_model is not None:
            imperfect_steering_angle = dynamics_model.predict(steering_angle)
        inference_time = time.perf_counter() - inference_start

        curvature = steering2curvature(math.degrees(steering_angle), LEXUS_WHEEL_BASE, LEXUS_STEERING_RATIO)

        step_start = time.perf_counter()
        vista_step(car, curvature, timings=timings)
        step_time = time.perf_counter() - step_start

        observation = grab_and_preprocess_obs(car, camera, antialias, resize_mode)

        vis_start = time.perf_counter()
        if save_video:
            vis_img = display.render()
            stream.write(vis_img[:, :, ::-1])
            stream_cropped.write(observation * 255.)
        vis_time = time.perf_counter() - vis_start

        print( f'\nStep {i_step} ({i_step / FPS:.0f}s) env step: {step_time:.2f}s | inference: {inference_time:.4f}s | visualization: {vis_time:.2f}s' )
        # for k, v_dict in timings.items():
        #     time_ = v_dict["time"]
        #     count = v_dict["count"]
        #     print(f'{k}: {time_:.2f}s ({time_/count:.2f}s per call)')

        i_step += 1
        if i_step % (FPS * LOG_FREQUENCY_SEC) == 0:
            print(f'Step {i_step} ({i_step / FPS:.0f}s) - Crashes so far: {len(crash_times)}')

        if check_out_of_lane(car):
            restart_at_frame = last_driven_frame_idx
            print(f'Crashed at step {i_step} (frame={last_driven_frame_idx}) ({i_step / FPS:.0f}s). Re-engaging at frame {restart_at_frame}!')
            crash_times.append(i_step / FPS)
            car.reset(0, 0, restart_at_frame)
            display.reset()
            if dynamics_model is not None:
                dynamics_model.reset()
            observation = grab_and_preprocess_obs(car, camera, antialias, resize_mode)
        if car.done:
            print(f'Finished at step {i_step} ({i_step / FPS:.0f}s).')
            break
        else:
            last_driven_frame_idx = car.frame_index

    if save_video:
        print('Saving trajectory to', video_name)
        stream.save(video_name)
        stream_cropped.save(video_name.replace('.avi', '_cropped.avi'))

    print(f'\nCrashes: {len(crash_times)}')

    return crash_times

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--antialias', action=argparse.BooleanOptionalAction, required=True, help='Use antialiasing when resizing the image')
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model.')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, required=True, help='Use Weights and Biases for logging.')
    parser.add_argument('--save-video', action=argparse.BooleanOptionalAction, required=True, help='Save video of model run.')
    parser.add_argument('--resize-mode', required=True, choices=['full', 'resize'], help='Resize mode of the input images (bags pre-processed for Vista).')
    parser.add_argument('--dynamics-model', default=None, help='Path to vehicle dynamics model (ONNX). IMPORTANT: ensure the model was trained on the same frequency as Vista is set up for. If not provided, the default (perfect) dynamics model will be used.')
    args = parser.parse_args()

    model = OnnxSteeringModel(args.model) # aquire GPU early (helpful for distributing runs across GPUs on a single machine)
    dynamics_model = OnnxDynamicsModel(args.dynamics_model) if args.dynamics_model is not None else None
    print(vars(args))

    # human-driven traces on the same track in different weather conditions
    trace_paths = []

    if args.resize_mode == 'full':
        trace_paths.extend([
            ['2021-10-26-10-49-06_e2e_rec_ss20_elva_eval_chunk-full_res', 'cloudy'],
            ['2021-10-26-11-08-59_e2e_rec_ss20_elva_back_eval_chunk-full_res', 'cloudy'],
        ])
    elif args.resize_mode == 'resize':
        trace_paths.extend([
            ['2021-10-26-10-49-06_e2e_rec_ss20_elva_eval_chunk-resize', 'cloudy'],
            ['2021-10-26-11-08-59_e2e_rec_ss20_elva_back_eval_chunk-resize', 'cloudy'],
        ])
    else:
        raise NotImplementedError(f'There is no such resize mode: {args.resize_mode}')

    if args.wandb:
        config = {
            'model_path': args.model,
            'trace_paths': trace_paths,
            'dynamics_model': args.dynamics_model,
            'antialias': args.antialias,
            'resize_mode': args.resize_mode,
            'save_video': args.save_video,
        }
        wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, config=config, job_type='vista-evaluation')

    trace_paths = [(os.path.join(TRACES_ROOT, track_path), condition) for track_path, condition in trace_paths]
    total_n_crashes = 0
    crashes_by_trace = defaultdict(int)
    crashes_by_condition = defaultdict(int)

    timestamp = int(time.time())
    model_name = os.path.basename(args.model).replace('.onnx', '')

    for trace, track_condition in trace_paths:
        output_video_name = f'{timestamp}_{os.path.basename(trace)}_{model_name}.avi'

        world = vista.World([trace], trace_config={'road_width': ROAD_WIDTH})
        car = world.spawn_agent(
            config={
                'length': LEXUS_LENGTH,
                'width': LEXUS_WIDTH,
                'wheel_base': LEXUS_WHEEL_BASE,
                'steering_ratio': LEXUS_STEERING_RATIO,
                'lookahead_road': False
            })

        camera_size = None
        if args.resize_mode == 'full':
            camera_size = (FULL_IMAGE_HEIGHT, FULL_IMAGE_WIDTH)
        elif args.resize_mode == 'downsample':
            camera_size = (FULL_IMAGE_HEIGHT // VISTA_DOWNSAMPLE_FACTOR, FULL_IMAGE_WIDTH // VISTA_DOWNSAMPLE_FACTOR)
        elif args.resize_mode == 'resize':
            camera_size = (FULL_IMAGE_HEIGHT // 5, FULL_IMAGE_WIDTH // 5)
        elif args.resize_mode == 'resize_and_crop':
            camera_size = (68, 264) # too hard to compute from the image size, just trust me

        camera = car.spawn_camera(config={'name': 'camera_front', 'size': camera_size})
        display = vista.Display(world, display_config={'gui_scale': 2, 'vis_full_frame': True })

        crash_times = run_episode(model, world, camera, car, antialias=args.antialias, 
                                                           save_video=args.save_video, 
                                                           video_name=output_video_name,
                                                           resize_mode=args.resize_mode,
                                                           dynamics_model=dynamics_model)
        crashes_by_trace[trace] = crash_times
        crashes_by_condition[track_condition] += len(crash_times)

        del camera._view_synthesis._renderer
        del camera
        del car
        del display
        del world

    total_crashes = sum([len(crash_times) for crash_times in crashes_by_trace.values()])

    print(f'\nCrashes by trace:')
    for trace, crash_times in crashes_by_trace.items():
        print(f'{trace}: {len(crash_times)}')
        for crash_time in crash_times:
            print(f'  > {crash_time:.0f}s')

    print(f'\nCrashes by condition:')
    for condition, n_crashes in crashes_by_condition.items():
        print(f'{condition}: {n_crashes}')

    print(f'Time spent: {time.time() - timestamp:.0f}s ({(time.time() - timestamp) / 60:.2f}min)')

    if args.wandb:
        condition_crash_counts = {f'crash_count_{condition}': count for condition, count in crashes_by_condition.items()}
        wandb.log({'crash_count': total_crashes, **condition_crash_counts})

    wandb.finish()
