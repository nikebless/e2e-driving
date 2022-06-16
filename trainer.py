import sys
from abc import abstractmethod
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

from tqdm.auto import tqdm
import wandb
import time

from metrics.metrics import calculate_open_loop_metrics, calculate_trajectory_open_loop_metrics

from ibc import optimizers
from pilotnet import PilotNetConditional, PilotnetControl, IbcPilotNet
from efficient_net import effnetv2_s

import train

class WeighedL1Loss(L1Loss):
    def __init__(self, weights):
        super().__init__(reduction='none')
        self.weights = weights

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = super().forward(input, target)
        return (loss * self.weights).mean()


class WeighedMSELoss(MSELoss):
    def __init__(self, weights):
        super().__init__(reduction='none')
        self.weights = weights

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = super().forward(input, target)
        return (loss * self.weights).mean()


class Trainer:

    def __init__(self, model_name=None, train_conf: train.TrainingConfig = None):  # todo:rename target_name->output_modality

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_name = train_conf.output_modality
        self.n_conditional_branches = train_conf.n_branches
        self.wandb_logging = False

        weights = torch.FloatTensor([(train_conf.loss_discount_rate ** i, train_conf.loss_discount_rate ** i)
                                     for i in range(train_conf.n_waypoints)]).to(self.device)
        weights = weights.flatten()
        if train_conf.n_branches > 1:  # todo: this is conditional learning specific and should be handled there
            weights = torch.cat(tuple(weights for i in range(train_conf.n_branches)), 0)

        if train_conf.loss == "mse":
            self.criterion = MSELoss()
        elif train_conf.loss == "mae":
            self.criterion = L1Loss()
        elif train_conf.loss == "mse-weighted":
            self.criterion = WeighedMSELoss(weights)
        elif train_conf.loss == "mae-weighted":
            self.criterion = WeighedL1Loss(weights)
        elif train_conf.loss == "ebm":
            self.criterion = CrossEntropyLoss(weights)
        else:
            print(f"Uknown loss function {train_conf.loss}")
            sys.exit()

        if train_conf.wandb_project:
            self.wandb_logging = True
            # wandb.init(project=wandb_project)

        if model_name:
            datetime_prefix = datetime.today().strftime('%Y%m%d%H%M%S')
            self.save_dir = Path("models") / f"{datetime_prefix}_{model_name}"
            self.save_dir.mkdir(parents=True, exist_ok=False)

    def force_cpu(self):
        self.device = 'cpu'

    def train(self, train_loader, valid_loader, n_epoch, patience=10, fps=30):
        model = self.model
        optimizer = self.optimizer
        criterion = self.criterion

        if self.wandb_logging:
            wandb.watch(model, criterion)

        best_valid_loss = float('inf')
        epochs_of_no_improve = 0

        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1, verbose=True)

        for epoch in range(n_epoch):

            progress_bar = tqdm(total=len(train_loader), smoothing=0)
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion, progress_bar, epoch)

            progress_bar.reset(total=len(valid_loader))
            valid_loss, predictions = self.evaluate(model, valid_loader, criterion, progress_bar, epoch, train_loss)

            scheduler.step(valid_loss)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

                torch.save(model.state_dict(), self.save_dir / f"best.pt")
                torch.save(model.state_dict(), self.save_dir / f"best-{epoch}.pt")
                epochs_of_no_improve = 0
                best_loss_marker = '*'
            else:
                epochs_of_no_improve += 1
                best_loss_marker = ''

            metrics = self.calculate_metrics(fps, predictions, valid_loader)
            # todo: this if elif is getting bad, abstract to separate classes
            if self.target_name == "steering_angle":
                whiteness = metrics['whiteness']
                mae = metrics['mae']
                left_mae = metrics['left_mae']
                straight_mae = metrics['straight_mae']
                right_mae = metrics['right_mae']
                progress_bar.set_description(f'{best_loss_marker}epoch {epoch + 1}'
                                             f' | train loss: {train_loss:.4f}'
                                             f' | valid loss: {valid_loss:.4f}'
                                             f' | whiteness: {whiteness:.4f}'
                                             f' | mae: {mae:.4f}'
                                             f' | l_mae: {left_mae:.4f}'
                                             f' | s_mae: {straight_mae:.4f}'
                                             f' | r_mae: {right_mae:.4f}')
            elif self.target_name == "waypoints":
                first_wp_mae = metrics['first_wp_mae']
                first_wp_whiteness = metrics['first_wp_whiteness']
                last_wp_mae = metrics['last_wp_mae']
                last_wp_whiteness = metrics['last_wp_whiteness']
                # frechet_distance = metrics['frechet_distance']
                progress_bar.set_description(f'{best_loss_marker}epoch {epoch + 1}'
                                             f' | train loss: {train_loss:.4f}'
                                             f' | valid loss: {valid_loss:.4f}'
                                             f' | 1_mae: {first_wp_mae:.4f}'
                                             f' | 1_whiteness: {first_wp_whiteness:.4f}'
                                             f' | last_mae: {last_wp_mae:.4f}'
                                             f' | last_whiteness: {last_wp_whiteness:.4f}')
                # f' | frechet: {frechet_distance:.4f}')

            if self.wandb_logging:
                metrics['epoch'] = epoch + 1
                metrics['train_loss'] = train_loss
                metrics['valid_loss'] = valid_loss
                wandb.log(metrics)

            if epochs_of_no_improve == patience:
                print(f'Early stopping, on epoch: {epoch + 1}.')
                break

        self.save_models(model, valid_loader)

        return best_valid_loss

    def calculate_metrics(self, fps, predictions, valid_loader):
        frames_df = valid_loader.dataset.frames
        if self.target_name == "steering_angle":
            true_steering_angles = frames_df.steering_angle.to_numpy()
            metrics = calculate_open_loop_metrics(predictions, true_steering_angles, fps=fps)

            left_turns = frames_df["turn_signal"] == 0  # TODO: remove magic values
            left_metrics = calculate_open_loop_metrics(predictions[left_turns], true_steering_angles[left_turns], fps=fps)
            metrics["left_mae"] = left_metrics["mae"]

            straight = frames_df["turn_signal"] == 1
            straight_metrics = calculate_open_loop_metrics(predictions[straight], true_steering_angles[straight], fps=fps)
            metrics["straight_mae"] = straight_metrics["mae"]

            right_turns = frames_df["turn_signal"] == 2
            right_metrics = calculate_open_loop_metrics(predictions[right_turns], true_steering_angles[right_turns], fps=fps)
            metrics["right_mae"] = right_metrics["mae"]

        elif self.target_name == "waypoints":
            true_waypoints = valid_loader.dataset.get_waypoints()
            metrics = calculate_trajectory_open_loop_metrics(predictions, true_waypoints, fps=fps)
        else:
            print(f"Uknown target name {self.target_name}")
            sys.exit()

        return metrics

    def save_models(self, model, valid_loader):
        torch.save(model.state_dict(), self.save_dir / "last.pt")
        if self.wandb_logging:
            wandb.save(f"{self.save_dir}/last.pt")
            wandb.save(f"{self.save_dir}/best.pt")

        self.save_onnx(model, valid_loader)

    def save_onnx(self, model, valid_loader):
        model.load_state_dict(torch.load(f"{self.save_dir}/best.pt"))
        model.to(self.device)

        data = iter(valid_loader).next()
        sample_inputs = self.create_onxx_input(data)
        torch.onnx.export(model, sample_inputs, f"{self.save_dir}/best.onnx")
        onnx.checker.check_model(f"{self.save_dir}/best.onnx")
        if self.wandb_logging:
            wandb.save(f"{self.save_dir}/best.onnx")

        model.load_state_dict(torch.load(f"{self.save_dir}/last.pt"))
        model.to(self.device)

        torch.onnx.export(model, sample_inputs, f"{self.save_dir}/last.onnx")
        onnx.checker.check_model(f"{self.save_dir}/last.onnx")
        if self.wandb_logging:
            wandb.save(f"{self.save_dir}/last.onnx")

    def create_onxx_input(self, data):
        return data[0]['image'].to(self.device)

    def train_epoch(self, model, loader, optimizer, criterion, progress_bar, epoch):
        running_loss = 0.0

        model.train()

        batch_wait_times = []

        ask_batch_timestamp = time.time()
        for i, (data, target_values, condition_mask) in enumerate(loader):
            recv_batch_timestap = time.time()
            batch_wait_time = recv_batch_timestap - ask_batch_timestamp
            batch_wait_times.append(batch_wait_time)
            print(f'Model waited for batch: {batch_wait_time * 1000:.2f}ms | avg: {np.mean(batch_wait_times) * 1000:.2f}ms (±{np.std(batch_wait_times) * 1000:.2f}) | max: {np.max(batch_wait_times) * 1000:.2f}ms | rate: {loader.batch_size / np.mean(batch_wait_times):.2f} FPS')

            optimizer.zero_grad()

            predictions, loss = self.train_batch(model, data, target_values, condition_mask, criterion)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            progress_bar.update(1)
            progress_bar.set_description(f'epoch {epoch+1} | train loss: {(running_loss / (i + 1)):.4f}')

            ask_batch_timestamp = time.time()

        return running_loss / len(loader)

    @abstractmethod
    def train_batch(self, model, data, target_values, condition_mask, criterion):
        pass

    @abstractmethod
    def predict(self, model, dataloader):
        pass

    def evaluate(self, model, iterator, criterion, progress_bar, epoch, train_loss):
        epoch_loss = 0.0
        model.eval()
        all_predictions = []

        with torch.no_grad():
            ask_batch_timestamp = time.time()
            for i, (data, target_values, condition_mask) in enumerate(iterator):
                recv_batch_timestap = time.time()
                print(f'Model waited for batch: %.2f ms', (recv_batch_timestap - ask_batch_timestamp) * 1000)

                predictions, loss = self.train_batch(model, data, target_values, condition_mask, criterion)
                epoch_loss += loss.item()
                all_predictions.extend(predictions.cpu().squeeze().numpy())

                progress_bar.update(1)
                progress_bar.set_description(f'epoch {epoch + 1} | train loss: {train_loss:.4f} | valid loss: {(epoch_loss / (i + 1)):.4f}')

                ask_batch_timestamp = time.time()

        total_loss = epoch_loss / len(iterator)
        result = np.array(all_predictions)
        return total_loss, result


class PilotNetTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        train_conf = kwargs['train_conf']

        self.model = effnetv2_s()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=train_conf.learning_rate, betas=(0.9, 0.999),
                                           eps=1e-08, weight_decay=train_conf.weight_decay, amsgrad=False)
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

    def predict(self, model, dataloader):
        all_predictions = []
        model.eval()

        with torch.no_grad():
            progress_bar = tqdm(total=len(dataloader), smoothing=0)
            progress_bar.set_description("Model predictions")
            for i, (data, target_values, condition_mask) in enumerate(dataloader):
                inputs = data['image'].to(self.device)
                predictions = model(inputs)
                all_predictions.extend(predictions.cpu().squeeze().numpy())
                progress_bar.update(1)

        return np.array(all_predictions)

    def train_batch(self, model, data, target_values, condition_mask, criterion):
        inputs = data['image'].to(self.device)
        target_values = target_values.to(self.device)
        predictions = model(inputs)
        return predictions, criterion(predictions, target_values)


class ControlTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        train_conf = kwargs['train_conf']

        self.model = PilotnetControl(train_conf.n_input_channels, train_conf.n_outputs)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=train_conf.learning_rate, betas=(0.9, 0.999),
                                           eps=1e-08, weight_decay=train_conf.weight_decay, amsgrad=False)
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

    def predict(self, model, dataloader):
        all_predictions = []
        model.eval()

        with torch.no_grad():
            progress_bar = tqdm(total=len(dataloader), smoothing=0)
            progress_bar.set_description("Model predictions")
            for i, (data, target_values, condition_mask) in enumerate(dataloader):
                inputs = data['image'].to(self.device)
                turn_signal = data['turn_signal']
                control = F.one_hot(turn_signal, 3).to(self.device)
                predictions = model(inputs, control)
                all_predictions.extend(predictions.cpu().squeeze().numpy())
                progress_bar.update(1)

        return np.array(all_predictions)

    def train_batch(self, model, data, target_values, condition_mask, criterion):
        inputs = data['image'].to(self.device)
        target_values = target_values.to(self.device)
        turn_signal = data['turn_signal']
        control = F.one_hot(turn_signal, 3).to(self.device)

        predictions = model(inputs, control)
        return predictions, criterion(predictions, target_values)

    def create_onxx_input(self, data):
        image_input = data[0]['image'].to(self.device)
        turn_signal = data[0]['turn_signal']
        control = F.one_hot(turn_signal, 3).to(torch.float32).to(self.device)
        return image_input, control


class ConditionalTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        train_conf = kwargs['train_conf']

        self.model = PilotNetConditional(train_conf.n_input_channels, train_conf.n_outputs, train_conf.n_branches)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=train_conf.learning_rate, betas=(0.9, 0.999),
                                           eps=1e-08, weight_decay=train_conf.weight_decay, amsgrad=False)
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

    def predict(self, model, dataloader):
        all_predictions = []
        model.eval()

        with torch.no_grad():
            progress_bar = tqdm(total=len(dataloader), smoothing=0)
            progress_bar.set_description("Model predictions")
            for i, (data, target_values, condition_mask) in enumerate(dataloader):
                inputs = data['image'].to(self.device)
                predictions = model(inputs)
                masked_predictions = predictions[condition_mask == 1]
                masked_predictions = masked_predictions.reshape(predictions.shape[0], -1)
                all_predictions.extend(masked_predictions.cpu().squeeze().numpy())
                progress_bar.update(1)

        return np.array(all_predictions)

    def train_batch(self, model, data, target_values, condition_mask, criterion):
        inputs = data['image'].to(self.device)
        target_values = target_values.to(self.device)
        condition_mask = condition_mask.to(self.device)

        predictions = model(inputs)

        loss = criterion(predictions*condition_mask, target_values) * self.n_conditional_branches

        masked_predictions = predictions[condition_mask == 1]
        return masked_predictions.reshape(predictions.shape[0], -1), loss


class IbcTrainer(Trainer):

    def __init__(self, train_dataloader, *args, **kwargs):
        super().__init__(*args, **kwargs)

        optim_config, stochastic_optim_config = self._initialize_config(kwargs['train_conf'], train_dataloader)

        self.model = IbcPilotNet()
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=optim_config.learning_rate,
            weight_decay=optim_config.weight_decay,
            betas=(optim_config.beta1, optim_config.beta2),
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=optim_config.lr_scheduler_step,
            gamma=optim_config.lr_scheduler_gamma,
        )

        self.stochastic_optimizer = optimizers.DerivativeFreeOptimizer.initialize(
            stochastic_optim_config,
            self.device,
        )

        self.steps = 0

    def _initialize_config(self, train_conf, train_dataloader):
        """Initialize train state based on config values."""

        optim_config = optimizers.OptimizerConfig(
            learning_rate=train_conf.learning_rate,
            weight_decay=train_conf.weight_decay,
        )

        target_bounds = train_dataloader.dataset.get_target_bounds()
        stochastic_optim_config = optimizers.DerivativeFreeConfig(
            bounds=target_bounds,
            train_samples=train_conf.stochastic_optimizer_train_samples,
        )

        return optim_config, stochastic_optim_config

    def predict(self, model, dataloader):
        all_predictions = []
        model.eval()

        with torch.no_grad():
            progress_bar = tqdm(total=len(dataloader), smoothing=0)
            progress_bar.set_description("Model predictions")
            for i, (data, target_values, condition_mask) in enumerate(dataloader):
                inputs = data['image'].to(self.device)
                predictions = model(inputs)
                all_predictions.extend(predictions.cpu().squeeze().numpy())
                progress_bar.update(1)

        return np.array(all_predictions)

    def train_batch(self, _, input, target, __, ___):
        inputs = input['image'].to(self.device)
        target = target.to(self.device, torch.float32)

        # Generate N negatives, one for each element in the batch: (B, N, D).
        negatives = self.stochastic_optimizer.sample(inputs.size(0), self.model)

        # Merge target and negatives: (B, N+1, D).
        targets = torch.cat([target.unsqueeze(dim=1), negatives], dim=1)

        # Generate a random permutation of the positives and negatives.
        permutation = torch.rand(targets.size(0), targets.size(1)).argsort(dim=1)
        targets = targets[torch.arange(targets.size(0)).unsqueeze(-1), permutation]

        # Get the original index of the positive. This will serve as the class label
        # for the loss.
        ground_truth = (permutation == 0).nonzero()[:, 1].view(-1, 1).to(self.device)

        # For every element in the mini-batch, there is 1 positive for which the EBM
        # should output a low energy value, and N negatives for which the EBM should
        # output high energy values.
        energy = self.model(inputs, targets)

        # Interpreting the energy as a negative logit, we can apply a cross entropy loss
        # to train the EBM.
        logits = -1.0 * energy

        print('\ninput to loss:')
        print('logits:', logits.shape)
        print('ground truth:', ground_truth.shape)
        loss = self.criterion(logits, ground_truth)

        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        self.steps += 1

        return energy, loss

    @torch.no_grad()
    def evaluate(self, iterator, _, progress_bar, epoch, train_loss):
        self.model.eval()
        all_predictions = []

        inference_times = []

        epoch_mse = 0.0
        ask_batch_timestamp = time.time()
        for i, (input, target) in enumerate(iterator):
            recv_batch_timestap = time.time()
            print(f'Model waited for batch: %.2f ms', (recv_batch_timestap - ask_batch_timestamp) * 1000)

            input = input.to(self.device)
            target = target.to(self.device)

            inference_start = time.time()
            preds = self.stochastic_optimizer.infer(input, self.model)
            inference_end = time.time()

            inference_time = inference_end - inference_start
            inference_times.append(inference_time)

            print(f'inference time: {inference_time} | avg : {np.mean(inference_times)} | max: {np.max(inference_times)} | min: {np.min(inference_times)}')

            mse = F.mse_loss(preds, target, reduction="none")
            epoch_mse += mse.mean(dim=-1).sum().item()

            all_predictions.extend(preds.cpu().squeeze().numpy())

            progress_bar.update(1)
            progress_bar.set_description(f'epoch {epoch + 1} | train loss: {train_loss:.4f} | valid loss: {(epoch_mse / (i + 1)):.4f}')

            ask_batch_timestamp = time.time()

        avg_mse = epoch_mse / len(iterator)
        result = np.array(all_predictions)
        return avg_mse, result
