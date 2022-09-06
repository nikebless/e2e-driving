import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import time
import numpy as np
import cv2
import torch
import math
import argparse
import wandb
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
WANDB_ENTITY = os.environ['WANDB_ENTITY']
WANDB_PROJECT = os.environ['WANDB_PROJECT']
TRACES_ROOT = os.path.join(BOLT_DIR, 'end-to-end', 'vista')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def vista_step(car, curvature=None, speed=None):
    if curvature is None: 
        curvature = car.trace.f_curvature(car.timestamp)
    if speed is None: 
        speed = car.trace.f_speed(car.timestamp)
    
    car.step_dynamics(action=np.array([curvature, speed]), dt=1/FPS)
    car.step_sensors()

def check_out_of_lane(car):
    distance_from_center = np.abs(car.relative_state.x)
    road_width = car.trace.road_width 
    half_road_width = road_width / 2
    return distance_from_center > half_road_width

def is_done_or_crashed(car): 
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

def run_episode(model, world, camera, car, save_video=False):
    stream = VideoStream(FPS)
    i_step = 0

    world.reset()
    display.reset()
    observation = grab_and_preprocess_obs(car, camera)

    while True:

        inference_start = time.perf_counter()
        model_input = np.moveaxis(observation, -1, 0).astype(np.float32)
        model_input = np.expand_dims(model_input, axis=0)
        predictions = model.predict(model_input)
        steering_angle = predictions[0]
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
            print(f'Step {i_step} ({i_step / FPS:.0f}s) - Still going...')

        if is_done_or_crashed(car):
            print(f'Crashed or Done at step {i_step} ({i_step / FPS:.0f}s)!')
            break

    if save_video:
        print("Saving trajectory...")
        stream.save(f"model_run.avi")

    print(f'\nReached {i_step} steps ({i_step / FPS:.0f}s)!')

    return i_step

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model.')
    parser.add_argument('--no-wandb', action='store_true', default=False, help='Do not use Weights and Biases for logging.')
    parser.add_argument('--save-video', action='store_true', default=False, help='Save video of model run.')
    args = parser.parse_args()

    config = {
        'model_path': args.model,
    }
    if not args.no_wandb:
        wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, config=config, job_type='vista-evaluation')

    print(vars(args))

    trace_paths = [
        "2022-08-31-15-37-37_elva_ebm_512_front",
    ]
    trace_paths = [os.path.join(TRACES_ROOT, p) for p in trace_paths]

    total_steps_completed = 0

    for trace in trace_paths:
        world = vista.World([trace], trace_config={'road_width': 4})
        car = world.spawn_agent(
            config={
                'length': LEXUS_LENGTH,
                'width': LEXUS_WIDTH,
                'wheel_base': LEXUS_WHEEL_BASE,
                'steering_ratio': LEXUS_STEERING_RATIO,
                'lookahead_road': False
            })
        camera = car.spawn_camera(config={'name': 'camera_front', 'size': (FULL_IMAGE_HEIGHT, FULL_IMAGE_WIDTH)})
        display = vista.Display(world, display_config={"gui_scale": 2, "vis_full_frame": True })
        model = OnnxModel(args.model)
        steps_completed = run_episode(model, world, camera, car, save_video=args.save_video)
        total_steps_completed += steps_completed

    if not args.no_wandb:
        wandb.log({'steps_completed': steps_completed})

    wandb.finish()
