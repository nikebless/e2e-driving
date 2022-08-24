import numpy as np

import pyvista as pv
from tqdm import tqdm
import glob
import argparse

CPOS = ((0, -400, 275), 
        (0, 50, 100), 
        (0, 0, 0))

TIME_STEP = 1

X_DIM = 0
Y_DIM = 1
Z_DIM = 2


def build_energy_mesh(energy, steering_dimension, time_start, time_end, invert_energy=False):

    time_dimension = np.linspace(time_start, time_end, time_end - time_start)

    X, Y = np.meshgrid(steering_dimension, time_dimension)
    Z = energy

    if invert_energy:
        Z = energy * -1

    grid = pv.StructuredGrid(X, Y, Z)
    grid['scalars'] = Z.ravel(order='F')

    return grid

def build_steering_arrow(energy_now, steering_values, offset):

    height = np.max(energy_now) - np.min(energy_now)
    steering_val = steering_values[np.argmin(energy_now)]
    floor = np.min(energy_now)
    center = [steering_val, offset, floor + height / 2]

    mesh = pv.CylinderStructured(radius=np.linspace(0, 1, 2), theta_resolution=10, height=height, direction=[0.001, 0.001, 0.5], center=center)
    return mesh, steering_val


if __name__ == '__main__':
    # TODO: 
    # 1. Show disengagements
    # 2. Show current frame #
    # 3. Optimize visualization
    #    - Re-order clipping/adding of energy slices

    # parse arguments
    parser = argparse.ArgumentParser(description='Visualize energy valley')
    parser.add_argument('--energy', type=str, default='energy_valley.npy', help='Path to energy file.')
    parser.add_argument('--samples', type=str, default='samples.npy', help='Path to numpy steering value samples file.')
    parser.add_argument('--output', type=str, default='energy_valley.mp4', help='Path to output file. Make sure to specify .mp4 extension.')
    parser.add_argument('--drive', type=str, default='2022*', help='Path to drive folder (could be glob pattern), which inside has a `front_wide` directory with PNG frames.')

    args = parser.parse_args()

    '''Invert steering angles so that source negative angles meant right turn'''
    invert_x = True

    time_window_size = 90
    pointer_offset = -10

    initial_time_start = -pointer_offset
    initial_time_end = time_window_size + pointer_offset

    files = sorted(glob.glob(f'{args.drive}/front_wide/*.png'))
    energy = np.load(args.energy)
    steering_angle_rads = np.load(args.samples).reshape(-1)
    steering_angle_degs = np.degrees(steering_angle_rads)

    if invert_x:
        steering_angle_degs = steering_angle_degs[::-1]

    sample_size = steering_angle_degs.shape[0]


    plotter = pv.Plotter(window_size=[1920//2, 608], off_screen=True)
    plotter.open_movie(args.output)
    plotter.show_bounds(show_zaxis=False, show_yaxis=False, xlabel='Steering Angle')

    cursor = initial_time_end - initial_time_start
    energy_slice = energy[:cursor]

    energy_mesh = build_energy_mesh(energy_slice, steering_angle_degs, initial_time_start, initial_time_end)
    steering_arrow, steering_val = build_steering_arrow(energy[0], steering_angle_degs, offset=-pointer_offset)

    plotter.add_mesh(energy_mesh, scalars='scalars', cmap='YlOrRd_r', lighting=True, show_scalar_bar=False)
    plotter.add_mesh(steering_arrow, scalars=np.repeat([1], len(steering_arrow.points)), cmap='winter_r', lighting=False, show_scalar_bar=False)
    plotter.add_background_image(files[0])

    plotter.show(cpos=CPOS, auto_close=False)
    plotter.write_frame()
    
    print('num files:', len(files))
    print('energy slices:', energy.shape[0])

    files = files[1:]


    for i in tqdm(range(energy.shape[0])):

        frame_img = files[i]

        try: plotter.remove_background_image()
        except: pass
        plotter.add_background_image(frame_img)

        cursor += 1
        energy_mesh.points[:, Y_DIM] -= TIME_STEP

        closest_point = energy_mesh.points[0]

        # when energy carpet gone too close to viewer, clip
        if closest_point[Y_DIM] < 0:
            # remove old energy slice
            tmp_points = energy_mesh.points.copy().reshape(sample_size, -1, 3)
            tmp_points = tmp_points[:, 1:, :]
            energy_mesh.points = tmp_points.reshape(-1, 3)

        # when energy carpet too short, add new slice
        if energy_mesh.points.shape[0] / sample_size < time_window_size:
            # add new energy frame
            tmp_points = energy_mesh.points.copy().reshape(sample_size, -1, 3)
            frame_energy_z = energy[cursor]
            frame_energy_x = steering_angle_degs
            frame_energy_y = tmp_points[:, -1, Y_DIM] + TIME_STEP
            frame_energy = np.stack([frame_energy_x, frame_energy_y, frame_energy_z], axis=1).reshape(-1, 1, 3)
            tmp_points = np.concatenate([tmp_points, frame_energy], axis=1)

            energy_mesh.points = tmp_points.reshape(-1, 3)
            energy_mesh.dimensions = (tmp_points.shape[1], *energy_mesh.dimensions[1:])
            energy_mesh.point_data['scalars'] = energy_mesh.points[:, Z_DIM].ravel(order='F')

        # 2. Update pointer
        steering_arrow_new, steering_val = build_steering_arrow(energy[i], steering_angle_degs, offset=-pointer_offset)
        steering_arrow.points = steering_arrow_new.points

        # plotter.add_text(f'Steering angle: {steering_val:.2f}', name='steering_angle')
        
        plotter.write_frame()


    
    plotter.close()