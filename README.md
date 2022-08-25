# Energy-Based Models for End-to-End Driving (Rally Estonia)

Train end-to-end driving models using Rally Estonia dataset.

Only lateral control (steering wheel) is predicted, longitudinal control (throttle) is not predicted and must be controlled using other means.

Implemented models:
- standard regression (MAE)
- energy-based regression ("implicit behavioral cloning")
- discretized classification
- Mixture Density Networks

## Dataset

To get access to the dataset, please fill out [this form](https://forms.gle/nDkwcpzgBoYeJBE39).

Before training a model, dataset needs to be downloaded and prepared . See [_dataloading_](./dataloading/README.md) manual for this. 

### Following input modalities are supported:
- Nvidia RGB cameras (only front wide camera used currently)

![camera_crop](./media/camera-crops.png "Camera crop")


## Training

### Environment setup

Environment can set up using conda by following commands:

```bash
# Set up Pytorch environment
conda create -n e2e pytorch torchvision cudatoolkit=11.1 jupyter pandas matplotlib tqdm scikit-learn scikit-image onnx seaborn -c pytorch -c nvidia
conda activate e2e

# Wandb
pip install wandb

# For visualising predictions
pip install opencv-contrib-python
# need to use specific version of moviepy as newer version did not work
pip install moviepy==1.0.0 
```

### Run training

Model can be trained using following command:

```bash
python train.py --input-modality nvidia-camera --output-modality steering_angle --patience 10 --max-epochs 100 --model-name steering-angle --model-type pilotnet-conditional --wandb-project summer-models-6 --dataset-folder <path to extracted dataset>
```

Use `--model-type` parameter to use different model architectures like `pilotnet`, `pilotnet-ebm`, `pilotnet-classifier` and `pilotnet-mdn`.

Use `--wandb-project` parameter to use log using W&B. To use without W&B, just omit this parameter. 

### Visualising results

To visualise trained models, use visualising scripts. Check out [visualising manual](./viz/README.md) for this.  

![visualisation](./media/visualisation.png "visualisation")

## Sub projects
- [_velocity_model_](velocity_model/README.md) - logitudinal model using predriven human trajectory
