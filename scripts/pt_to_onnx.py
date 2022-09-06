# /home/nikita/e2e-driving/_models/ibc_pure_last.pt

import logging
from pathlib import Path
import argparse

import torch
import onnx

from dataloading.nvidia import NvidiaValidationDataset
from ebm import optimizers
from pilotnet import PilotNet, PilotnetEBM, PilotnetClassifier, PilotnetMDN
from classifier.infer import ClassifierInferrer
from mdn.infer import MDNInferrer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_model(model_path, model_type, device, config={}):
    inference_config = None
    inference_wrapper = None

    if model_type == 'pilotnet':
        model = PilotNet()
    elif model_type == 'pilotnet-ebm':
        n_samples = config['n_samples']
        n_dfo_iters = config.get('ebm_dfo_iters', 0)
        ebm_constant_samples = config.get('ebm_constant_samples', True)
        steering_bound = config['steering_bound']
        
        inference_config = optimizers.DerivativeFreeConfig(
            bound=steering_bound,
            train_samples=n_samples,
            inference_samples=n_samples,
            iters=n_dfo_iters,
        )
        inference_wrapper = optimizers.DFOptimizerConst if ebm_constant_samples else optimizers.DFOptimizer
        model = PilotnetEBM()
        model = inference_wrapper(model, inference_config)
    elif model_type == 'pilotnet-classifier':
        steering_bound = config['steering_bound']
        weights = torch.load(model_path, map_location=torch.device('cpu'))
        for _, weight in weights.items(): pass # access output layer weights
        output_layer_size = weight.shape[0]
        model = PilotnetClassifier(output_layer_size)
        model = ClassifierInferrer(model, steering_bound, output_layer_size)
    elif model_type == 'pilotnet-mdn':
        n_gaussians = config['n_gaussians']
        model = PilotnetMDN(n_gaussians=n_gaussians)
        model = MDNInferrer(model)
    else:
        raise ValueError(f'Unknown model type: {model_type}')

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    return model

def convert_pt_to_onnx(model_path, model_type, output_path=None, config={}):
    model = initialize_model(model_path, model_type, device, config)
    data_loader = get_loader(batch_size=1)
    inputs, _, _ = iter(data_loader).next()
    frames = inputs['image'].to(device)
    sample_inputs = [frames]
    input_names = ['x']
    dynamic_axes = None
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


def get_loader(batch_size=1, dataset_path='/data/Bolt/end-to-end/rally-estonia-cropped'):
    num_workers = 2

    validset = NvidiaValidationDataset(Path(dataset_path), group_size=1)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False,
                                            num_workers=num_workers, pin_memory=True,
                                            persistent_workers=False)
    return valid_loader


def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--ebm-constant-samples', default=True, action='store_true', help='Use constant samples instead of random samples for EBM.')
    parser.add_argument('--file', type=str, help='Path to the PyTorch model')
    parser.add_argument('--model-type', type=str, help='Type of the model.', choices=['pilotnet-ebm', 'pilotnet-classifier'])
    parser.add_argument('--ebm-dfo-iters', default=0, type=int, help='Number of DFO iterations. Ignored if --with_dfo is not set.')
    parser.add_argument('--n-gaussians', default=3, type=int, help='Number of gaussians for an MDN.')
    parser.add_argument('--n-samples', default=128, type=int, help='Number of action samples in the discretization/as input to EBM.')
    parser.add_argument('--output', default=None, type=str, help='Path to the output ONNX model')
    parser.add_argument('--steering-bound', default=4.5, type=float, help='Bound for the steering angle, in radians. If not set, the model will use the default bound.')
    parser.add_argument('--verbose', default=False, action='store_true', help='Print debug messages')

    args = parser.parse_args(raw_args)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

   

    output_path = convert_pt_to_onnx(args.file, args.model_type, args.output, vars(args))
    print(f'Successfuly converted to ONNX: {output_path}')


if __name__ == '__main__':
    main()