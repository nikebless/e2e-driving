from collections import defaultdict
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['EGL_DEVICE_ID'] = '0'

import time
import numpy as np
import cv2
import math
import argparse
import wandb
import traceback
import dotenv
dotenv.load_dotenv()

import vista
from vista.entities.agents.Dynamics import steering2curvature

from common import BOLT_DIR, LEXUS_LENGTH, LEXUS_WIDTH, LEXUS_WHEEL_BASE, LEXUS_STEERING_RATIO, \
                   FULL_IMAGE_HEIGHT, FULL_IMAGE_WIDTH, \
                   IMAGE_CROP_XMIN, IMAGE_CROP_XMAX, IMAGE_CROP_YMIN, IMAGE_CROP_YMAX, \
                   OnnxModel
from vista_sim.video import VideoStream


LOG_FREQUENCY_SEC = 1
FPS = 13
WANDB_ENTITY = os.environ.get('WANDB_ENTITY', None)
WANDB_PROJECT = os.environ.get('WANDB_PROJECT', None)
TRACES_ROOT = os.path.join(BOLT_DIR, 'end-to-end', 'vista')
FRAME_START_OFFSET = 100
ROAD_WIDTH = 3.5
MAX_OPENGL_RETRIES = int(os.environ.get('MAX_OPENGL_RETRIES', 10))


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
    while not step_sensors_safe(car) or n_attempts > MAX_OPENGL_RETRIES:
        n_attempts += 1
        print(f'Waiting 5 sec before another step_sensors() attempt... (tried {n_attempts}/{MAX_OPENGL_RETRIES} times)')
        time.sleep(5)


def check_out_of_lane(car):
    distance_from_center = np.abs(car.relative_state.x)
    road_width = car.trace.road_width 
    half_road_width = road_width / 2
    return distance_from_center > half_road_width

def is_crashed(car): 
    return check_out_of_lane(car) or car.done

def resize(img):
    scale = 0.2
    height = IMAGE_CROP_YMAX - IMAGE_CROP_YMIN
    width = IMAGE_CROP_XMAX - IMAGE_CROP_XMIN

    scaled_width = int(width * scale)
    scaled_height = int(height * scale)

    return cv2.resize(img, dsize=(scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)

def normalise(img):
    return (img / 255.0)

def crop(img):
    return img[IMAGE_CROP_YMIN:IMAGE_CROP_YMAX, IMAGE_CROP_XMIN:IMAGE_CROP_XMAX, :]

def preprocess(full_obs):
    img = crop( full_obs )
    img = resize( img )
    img = normalise( img )
    return img

def grab_and_preprocess_obs(car, camera):
    full_obs = car.observations[camera.name]
    obs = preprocess(full_obs)
    return obs

def run_episode(model, world, camera, car, save_video=False, video_name='model_run.avi'):
    stream = VideoStream(FPS)
    i_step = 0

    world.reset()
    car.reset(0, 0, FRAME_START_OFFSET)
    display.reset()
    observation = grab_and_preprocess_obs(car, camera)

    last_driven_frame_idx = 0
    crash_count = 0

    while True:

        inference_start = time.perf_counter()
        model_input = np.moveaxis(observation, -1, 0).astype(np.float32)
        model_input = np.expand_dims(model_input, axis=0)
        predictions = model.predict(model_input)
        steering_angle = predictions.item()
        inference_time = time.perf_counter() - inference_start

        curvature = steering2curvature(math.degrees(steering_angle), LEXUS_WHEEL_BASE, LEXUS_STEERING_RATIO)

        step_start = time.perf_counter()
        vista_step(car, curvature)
        step_time = time.perf_counter() - step_start

        observation = grab_and_preprocess_obs(car, camera)

        vis_start = time.perf_counter()
        if save_video:
            vis_img = display.render()
            stream.write(vis_img[:, :, ::-1])
        vis_time = time.perf_counter() - vis_start

        print( f'dynamics step: {step_time:.2f}s | inference: {inference_time:.4f}s | visualization: {vis_time:.2f}s' )

        i_step += 1
        if i_step % (FPS * LOG_FREQUENCY_SEC) == 0:
            print(f'Step {i_step} ({i_step / FPS:.0f}s) - Crashes so far: {crash_count}')

        if check_out_of_lane(car):
            restart_at_frame = last_driven_frame_idx
            print(f'Crashed at step {i_step} (frame={last_driven_frame_idx}) ({i_step / FPS:.0f}s). Re-engaging at frame {restart_at_frame}!')
            crash_count += 1
            car.reset(0, 0, restart_at_frame)
            display.reset()
            observation = grab_and_preprocess_obs(car, camera)
        if car.done:
            print(f'Finished at step {i_step} ({i_step / FPS:.0f}s).')
            break
        else:
            last_driven_frame_idx = car.frame_index

    if save_video:
        print('Saving trajectory to', video_name)
        stream.save(video_name)

    print(f'\nCrashes: {crash_count}')

    return crash_count

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model.')
    parser.add_argument('--no-wandb', action='store_true', default=False, help='Do not use Weights and Biases for logging.')
    parser.add_argument('--save-video', action='store_true', default=False, help='Save video of model run.')
    args = parser.parse_args()

    model = OnnxModel(args.model) # aquire GPU early (helpful for distributing runs across GPUs on a single machine)

    print(vars(args))

    # human-driven traces on the same track in different weather conditions
    trace_paths = [
        # ['2022-06-10-13-23-01_e2e_elva_forward_4_3_km_section', 'sunny'], 
        # ['2022-06-10-13-03-20_e2e_elva_backward_4_3_km_section', 'sunny'],
        ['2021-10-26-10-49-06_e2e_rec_ss20_elva_eval_chunk', 'cloudy'],
        # ['2021-10-26-11-08-59_e2e_rec_ss20_elva_back_eval_chunk', 'cloudy'],
    ]

    if not args.no_wandb:
        config = {
            'model_path': args.model,
            'trace_paths': trace_paths,
        }
        wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, config=config, job_type='vista-evaluation')

    trace_paths = [(os.path.join(TRACES_ROOT, track_path), condition) for track_path, condition in trace_paths]
    total_n_crashes = 0
    crashes_by_trace = defaultdict(int)
    crashes_by_condition = defaultdict(int)

    timestamp = int(time.time())

    for trace, track_condition in trace_paths:
        output_video_name = f'{timestamp}_{os.path.basename(trace)}.avi'

        world = vista.World([trace], trace_config={'road_width': ROAD_WIDTH})
        car = world.spawn_agent(
            config={
                'length': LEXUS_LENGTH,
                'width': LEXUS_WIDTH,
                'wheel_base': LEXUS_WHEEL_BASE,
                'steering_ratio': LEXUS_STEERING_RATIO,
                'lookahead_road': False
            })
        camera = car.spawn_camera(config={'name': 'camera_front', 'size': (FULL_IMAGE_HEIGHT, FULL_IMAGE_WIDTH)})
        display = vista.Display(world, display_config={'gui_scale': 2, 'vis_full_frame': True })

        n_crashes = run_episode(model, world, camera, car, save_video=args.save_video, video_name=output_video_name)
        crashes_by_trace[trace] = n_crashes
        crashes_by_condition[track_condition] += n_crashes

    total_crashes = sum(crashes_by_trace.values())

    print(f'\nCrashes by trace:')
    for trace, n_crashes in crashes_by_trace.items():
        print(f'{trace}: {n_crashes}')

    print(f'\nCrashes by condition:')
    for condition, n_crashes in crashes_by_condition.items():
        print(f'{condition}: {n_crashes}')

    if not args.no_wandb:
        wandb.log({'crash_count': total_crashes})

    wandb.finish()
