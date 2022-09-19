from pathlib import Path
import os
import argparse
import pandas as pd
import wandb

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, 'scripts', 'cache')

import dotenv
dotenv.load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

WANDB_ENTITY = os.getenv('WANDB_ENTITY')
WANDB_PROJECT = os.getenv('WANDB_PROJECT')


def evaluate_model_on_elva(model_name):
    import dataloading.nvidia as nv
    from common import OnnxModel, BOLT_DIR
    import torchvision.transforms as transforms
    import torch
    from metrics.metrics import calculate_open_loop_metrics
    from tqdm import tqdm
    import numpy as np


    path_to_model = os.path.join(PROJECT_ROOT, '_models', model_name + '.onnx')
    model = OnnxModel(path_to_model)

    dataset = nv.NvidiaElvaDataset(Path(os.path.join(BOLT_DIR, 'end-to-end/drives-ebm-paper')), eval_section=True, group_size=1, transform=transforms.Compose([nv.NvidiaCropWide(), nv.Normalize()]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, collate_fn=dataset.collate_fn)

    predictions = []

    for i, (input, target, _) in enumerate(tqdm(loader)):
        inputs = input['image'].numpy()
        target = target.numpy().astype(np.float32)

        ort_outs = model.predict(inputs)
        pred_angle = np.atleast_2d(ort_outs)[0,0]
        predictions.append(pred_angle)

    predictions = np.array(predictions)
    true_steering_angles = loader.dataset.frames.steering_angle.to_numpy()
    assert len(predictions) == len(true_steering_angles), f'len(predictions)={len(predictions)} != len(true_steering_angles)={len(true_steering_angles)}'

    metrics = calculate_open_loop_metrics(predictions, true_steering_angles, fps=30)

    return metrics

def schedule_run(model_name):
    os.system(f'sbatch scripts/offline_elva_eval_hpc.sh --model {model_name}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, required=False, help='Name of model to evaluate.')

    args = parser.parse_args()

    if args.model is None:
        df = pd.read_csv(os.path.join(PROJECT_ROOT, 'notebooks', 'ebm-experiments.csv'))
        os.makedirs(CACHE_DIR, exist_ok=True)

        api = wandb.Api()
        runs = api.runs(f'{WANDB_ENTITY}/{WANDB_PROJECT}')
        done_models = set()
        for run in runs:
            if run.state == 'finished' and 'offline-elva-evaluation' in run.tags and run.summary.get('mae', None) is not None:
                done_models.add(run.config.get('model_path'))

        model_names = set(df['model_name'].unique().tolist()) - done_models
        for model_name in model_names:
            schedule_run(model_name)
    else:
        model_name = args.model
        config = {
            'model_path': model_name,
        }
        wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=model_name, job_type='offline-elva-evaluation', tags=['offline-elva-evaluation'], config=config)
        print('config:', config)

        metrics = evaluate_model_on_elva(model_name)

        wandb.log(metrics)
        wandb.finish()
