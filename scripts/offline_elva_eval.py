import dataloading.nvidia as nv
from common import OnnxModel
import torchvision.transforms as transforms
import pandas as pd
from pathlib import Path
import torch
from tqdm import tqdm
from metrics.metrics import calculate_open_loop_metrics
import numpy as np
import os
import wandb

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, 'scripts', 'cache')

import dotenv
dotenv.load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

WANDB_ENTITY = os.getenv('WANDB_ENTITY')
WANDB_PROJECT = os.getenv('WANDB_PROJECT')


def evaluate_model_on_elva(model_name):
    path_to_model = os.path.join(PROJECT_ROOT, '_models', model_name + '.onnx')
    model = OnnxModel(path_to_model)

    dataset = nv.NvidiaElvaDataset(Path('/data/Bolt/end-to-end/drives-ebm-paper'), eval_section=True, group_size=1, transform=transforms.Compose([nv.NvidiaCropWide(), nv.Normalize()]))
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

if __name__ == '__main__':

    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'notebooks', 'ebm-experiments.csv'))
    os.makedirs(CACHE_DIR, exist_ok=True)

    for index, row in df.iterrows():
        model_name = row['model_name']
        metrics_cache = Path(os.path.join(CACHE_DIR, f'elva_metrics_{model_name}.csv'))
        if Path(metrics_cache).exists():
            print(f'Already evaluated {model_name}, skipping')
            continue
        else:
            print('Evaluating', model_name)

        config = {
            'model_path': model_name,
        }
        wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=model_name, job_type='offline-elva-evaluation', tags=['offline-elva-evaluation'], config=config)

        metrics = evaluate_model_on_elva(model_name)
        metrics_df = pd.DataFrame(metrics, index=[0])
        metrics_df.to_csv(metrics_cache, index=False)
        wandb.log(metrics)
        wandb.finish()
