import os
import sys
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['EGL_DEVICE_ID'] = os.environ.get('CUDA_AVAILABLE_DEVICES', '0')

import uuid
import time
from datetime import datetime
from collections import defaultdict
import argparse
import traceback
import dotenv
dotenv.load_dotenv()

import wandb
import numpy as np
import math

import vista
from vista.entities.agents.Dynamics import steering2curvature
from vista.entities.sensors.camera_utils.ViewSynthesis import DepthModes

from common import BOLT_DIR, OnnxSteeringModel, Timing
from vista_sim.preprocessing import grab_and_preprocess_obs, get_camera_size
from vista_sim.video_stream import VideoStream
from vista_sim.dynamics_model import OnnxDynamicsModel
from vista_sim.car_constants import LEXUS_LENGTH, LEXUS_WIDTH, LEXUS_WHEEL_BASE, LEXUS_STEERING_RATIO


WANDB_ENTITY = os.environ.get('WANDB_ENTITY', None)
WANDB_PROJECT = os.environ.get('WANDB_PROJECT', None)
TRACES_ROOT = os.path.join(BOLT_DIR, 'end-to-end', 'vista')
MAX_OPENGL_RETRIES = int(os.environ.get('MAX_OPENGL_RETRIES', 10))
OUTPUT_DIR = 'out'

LOG_FREQUENCY_SEC = 1
SRC_FPS = 30
FPS = 10
FRAME_START_OFFSET = 100
SECONDS_SKIP_AFTER_CRASH = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)


def step_sensors_safe(car):
    successful = False
    try: 
        car.step_sensors()
        successful = True
    except Exception as err:
        print(err)
        traceback.print_exc()
    return successful

def vista_step(car, curvature=None, speed=None):
    if curvature is None: 
        curvature = car.trace.f_curvature(car.timestamp)
    if speed is None: 
        speed = car.trace.f_speed(car.timestamp)
    
    car.step_dynamics(action=np.array([curvature, speed]), dt=1/FPS)

    n_attempts = 0
    while not step_sensors_safe(car) and n_attempts < MAX_OPENGL_RETRIES:
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


def run_evaluation_episode(model, world, camera, car, antialias, video_dir, save_video=False, resize_mode='full', dynamics_model=None):
    if save_video:
        stream = VideoStream(video_dir, FPS, suffix='_full')
        stream_cropped = VideoStream(video_dir, FPS, suffix='_cropped', no_encoding=True)

    i_step = 0
    i_segment = 0

    world.reset()
    # FRAME_START_OFFSET is magic. without doing this, restarting at an earlier frame started the trace EARLIER than frame 0. no idea how this works.
    # TODO: check that this is still necessary with new traces
    car.reset(0, i_segment, FRAME_START_OFFSET) 
    display.reset()
    observation = grab_and_preprocess_obs(car, camera, antialias, resize_mode)
    n_segments = len(car.trace.good_timestamps[car.trace._multi_sensor.master_sensor])

    last_driven_frame_idx = 0
    crash_times = []


    while True:

        inference_start = time.perf_counter()
        model_input = np.moveaxis(observation, -1, 0)
        model_input = np.expand_dims(model_input, axis=0)
        predictions = model.predict(model_input)
        steering_angle = predictions.item()
        if dynamics_model is not None:
            steering_angle = dynamics_model.predict(steering_angle)
        inference_time = time.perf_counter() - inference_start

        curvature = steering2curvature(math.degrees(steering_angle), LEXUS_WHEEL_BASE, LEXUS_STEERING_RATIO)

        step_start = time.perf_counter()
        vista_step(car, curvature)
        step_time = time.perf_counter() - step_start

        observation = grab_and_preprocess_obs(car, camera, antialias, resize_mode)

        vis_start = time.perf_counter()
        if save_video:
            vis_img = display.render()
            stream.write(vis_img[:, :, ::-1])
            stream_cropped.write(observation * 255.)
        vis_time = time.perf_counter() - vis_start

        print( f'\nStep {i_step} ({i_step / FPS:.0f}s) env step: {step_time:.2f}s | inference: {inference_time:.4f}s | visualization: {vis_time:.2f}s' )

        i_step += 1
        if i_step % (FPS * LOG_FREQUENCY_SEC) == 0:
            print(f'Step {i_step} ({i_step / FPS:.0f}s) - Crashes so far: {len(crash_times)}')

        if check_out_of_lane(car):
            restart_at_frame = last_driven_frame_idx + SECONDS_SKIP_AFTER_CRASH*SRC_FPS
            print(f'Crashed at step {i_step} (frame={last_driven_frame_idx}) ({i_step / FPS:.0f}s). Re-engaging at frame {restart_at_frame}!')
            crash_times.append(i_step / FPS)
            car.reset(0, i_segment, restart_at_frame)
            display.reset()
            if dynamics_model is not None:
                dynamics_model.reset()
            observation = grab_and_preprocess_obs(car, camera, antialias, resize_mode)
        if car.done:
            if i_segment < n_segments - 1:
                # only finished segment, not the whole trace

                print(f'Finished segment {i_segment} at step ({i_step}) ({i_step / FPS:.0f}s).')
                i_segment += 1
                car.reset(0, i_segment, FRAME_START_OFFSET)
                display.reset()
                if dynamics_model is not None:
                    dynamics_model.reset()
                observation = grab_and_preprocess_obs(car, camera, antialias, resize_mode)
                last_driven_frame_idx = car.frame_index

            else:
                print(f'Finished trace at step {i_step} ({i_step / FPS:.0f}s).')
                break
        else:
            last_driven_frame_idx = car.frame_index

    if save_video:
        print('Saving trace videos to:', video_dir)
        stream.save(os.path.join(video_dir, 'full.avi'))
        stream_cropped.save(os.path.join(video_dir, 'cropped.avi'))

    print(f'\nCrashes: {len(crash_times)}')

    return crash_times

if __name__ == '__main__':

    run_start_time = int(time.time())

    parser = argparse.ArgumentParser()

    if sys.version_info[1] < 9:
        parser.add_argument('--antialias', action='store_true', help='Use antialiasing when resizing the image')
        parser.add_argument('--wandb',action='store_true', help='Use Weights and Biases for logging.')
        parser.add_argument('--save-video', action='store_true', help='Save video of model run.')
    else:
        # this will require supplying either --antialias or --no-antialias
        parser.add_argument('--antialias', action=argparse.BooleanOptionalAction, required=True, help='Use antialiasing when resizing the image')
        parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, required=True, help='Use Weights and Biases for logging.')
        parser.add_argument('--save-video', action=argparse.BooleanOptionalAction, required=True, help='Save video of model run.')

    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model.')
    parser.add_argument('--resize-mode', required=True, choices=['full', 'resize'], help='Resize mode of the input images (bags pre-processed for Vista).')
    parser.add_argument('--dynamics-model', default=None, help='Path to vehicle dynamics model (ONNX). IMPORTANT: ensure the model was trained on the same frequency as Vista is set up for. If not provided, the default (perfect) dynamics model will be used.')
    parser.add_argument('--road-width', type=float, default=2.5, help='Vista road width in meters.')
    parser.add_argument('--comment', type=str, default=None, help='Run description.')
    parser.add_argument('--depth-mode', type=str, default='monodepth', choices=['fixed_plane', 'monodepth'], help='''Depth approximation mode. Monodepth uses a neural network to estimate depth from a single image, 
                                                                                                                     resulting in fewer artifacts in synthesized images. Fixed plane uses a fixed plane at a fixed distance from the camera.''')
    parser.add_argument('--traces', type=str, nargs='+', default=None, help='Traces to evaluate on. If not provided, a human-driven Elva track from late October 2021 will be used.')
    args = parser.parse_args()
    print(vars(args))

    model = OnnxSteeringModel(args.model) # aquire GPU early (helpful for distributing runs across GPUs on a single machine)
    dynamics_model = OnnxDynamicsModel(args.dynamics_model) if args.dynamics_model is not None else None

    trace_paths = []
    if args.traces is None:
        if args.resize_mode == 'full':
            trace_paths.extend([
                '2021-10-26-10-49-06_e2e_rec_ss20_elva_eval_chunk-full_res',
                '2021-10-26-11-08-59_e2e_rec_ss20_elva_back_eval_chunk-full_res',
            ])
        elif args.resize_mode == 'resize':
            trace_paths.extend([
                '2021-10-26-10-49-06_e2e_rec_ss20_elva_eval_chunk-resize',
                '2021-10-26-11-08-59_e2e_rec_ss20_elva_back_eval_chunk-resize',
            ])
        else:
            raise NotImplementedError(f'There is no such resize mode: {args.resize_mode}')
    else:
        for trace_path in args.traces:
            trace_paths.append(trace_path)

    if args.wandb:
        config = {
            'model_path': args.model,
            'trace_paths': trace_paths,
            'dynamics_model': args.dynamics_model,
            'antialias': args.antialias,
            'resize_mode': args.resize_mode,
            'save_video': args.save_video,
            'road_width': args.road_width,
            'depth_mode': args.depth_mode,
        }
        wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, config=config, job_type='vista-evaluation', notes=args.comment)

    trace_paths = [os.path.join(TRACES_ROOT, track_path) for track_path in trace_paths]
    total_n_crashes = 0
    crashes_by_trace = defaultdict(int)

    model_name = os.path.basename(args.model).replace('.onnx', '')
    date_time_str = datetime.fromtimestamp(run_start_time).replace(microsecond=0).isoformat()
    unique_chars = uuid.uuid4().hex[:3] # uuid4 is robust to same-node time collisions
    run_dir = os.path.join(OUTPUT_DIR, f'{date_time_str}_{unique_chars}_{model_name}')

    for trace in trace_paths:
        run_trace_dir = os.path.join(run_dir, os.path.basename(trace))
        os.makedirs(run_trace_dir) # will fail if the directory already exists

        world = vista.World([trace], trace_config={'road_width': args.road_width})
        car = world.spawn_agent(
            config={
                'length': LEXUS_LENGTH,
                'width': LEXUS_WIDTH,
                'wheel_base': LEXUS_WHEEL_BASE,
                'steering_ratio': LEXUS_STEERING_RATIO,
                'lookahead_road': False
            })

        camera_size = get_camera_size(args.resize_mode)
        camera = car.spawn_camera(config={'name': 'camera_front', 'size': camera_size, 'depth_mode': DepthModes.MONODEPTH})
        display = vista.Display(world, display_config={'gui_scale': 2, 'vis_full_frame': True })

        crash_times = run_evaluation_episode(model, world, camera, car, antialias=args.antialias, 
                                                           save_video=args.save_video, 
                                                           video_dir=run_trace_dir,
                                                           resize_mode=args.resize_mode,
                                                           dynamics_model=dynamics_model)
        crashes_by_trace[trace] = crash_times

        # cleanup
        del camera._view_synthesis._renderer
        del camera
        del car
        del display
        del world

    print(f'\nCrashes by trace:')
    for trace, crash_times in crashes_by_trace.items():
        print(f'{trace}: {len(crash_times)}')
        for crash_time in crash_times:
            print(f'  > {crash_time:.0f}s')

    print(f'Time spent: {time.time() - run_start_time:.0f}s ({(time.time() - run_start_time) / 60:.2f}min)')

    if args.wandb:
        total_crashes = sum([len(crash_times) for crash_times in crashes_by_trace.values()])
        wandb.log({'crash_count': total_crashes })

    wandb.finish()
