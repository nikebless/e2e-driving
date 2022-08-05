# /home/nikita/e2e-driving/_models/ibc_pure_last.pt

import logging
from pathlib import Path
import argparse

import torch
import onnx
import numpy as np


from ibc import optimizers
from dataloading.nvidia import NvidiaValidationDataset
from pilotnet import PilotnetEBM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convert_pt_to_onnx(model_path, batch_size, output_path, with_choice, n_samples, iters, args):
    data_loader = get_loader(batch_size=batch_size)

    model = PilotnetEBM()
    if with_choice:
        stochastic_optim_config = optimizers.DerivativeFreeConfig(
            bounds=torch.tensor([[-args['steering_bound']], [args['steering_bound']]]),
            train_samples=0,
            inference_samples=n_samples,
            iters=iters,
        )
        inference_wrapper = optimizers.DFOptimizerConst if args['use_constant_samples'] else optimizers.DFOptimizer
        model = inference_wrapper(model, stochastic_optim_config, batch_size=batch_size)

    model.load_state_dict(torch.load(model_path))
    model.to(device)

    inputs, _, _ = iter(data_loader).next()

    frames = inputs['image'].to(device)
    sample_inputs = [frames]

    input_names = ['x']
    
    dynamic_axes = None

    if not with_choice:
        random_actions = torch.tensor(np.random.uniform(-args['steering_bound'], args['steering_bound'], size=(batch_size, n_samples, 1)), device=device, dtype=torch.float32)
        sample_inputs.append(random_actions)
        input_names.append('y')

    output_path = Path(model_path).with_suffix('.onnx') if output_path is None else output_path

    sample_inputs.append({}) # onnx export magic

    torch.onnx.export(model, 
                      sample_inputs, 
                      output_path, 
                      input_names=input_names, 
                      dynamic_axes=dynamic_axes, 
                      do_constant_folding=False,
                      opset_version=9,
                      )
    onnx.checker.check_model(str(output_path))
    return str(output_path)


def get_loader(batch_size=1, dataset_path='/data/Bolt/dataset-new-small/summer2021', output_modality='steering_angle'):
    n_branches = 1
    num_workers = 2

    validset = NvidiaValidationDataset(Path(dataset_path), output_modality, n_branches, n_waypoints=1, group_size=1)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False,
                                            num_workers=num_workers, pin_memory=True,
                                            persistent_workers=False)
    return valid_loader


def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='Path to the PyTorch model')
    parser.add_argument('--output', type=str, help='Path to the output ONNX model')
    parser.add_argument('--with-choice', default=False, action='store_true', help='Choose the steering angle in the ONNX graph. If false, simply output energies for input actions.')
    parser.add_argument('--samples', default=1024, type=int, help='Number of action samples.')
    parser.add_argument('--iters', default=3, type=int, help='Number of DFO iterations. Ignored if --with_dfo is not set.')
    parser.add_argument('--bs', default=1, type=int, help='Batch size. Necessary when --with_choice is NOT set, inference will only be available with this batch size.')
    parser.add_argument('--verbose', default=False, action='store_true', help='Print debug messages')
    parser.add_argument('--steering-bound', default=4.5, type=float, help='Bounds for the steering angle, in radians. If not set, the model will use the default bounds.')
    parser.add_argument('--use-constant-samples', default=False, action='store_true', help='Use constant samples instead of random samples.')

    args = parser.parse_args(raw_args)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    output_path = convert_pt_to_onnx(args.file, args.bs, args.output, args.with_choice, args.samples, args.iters, vars(args))

    print(f'Successfuly converted to ONNX: {output_path}')


if __name__ == '__main__':
    main()