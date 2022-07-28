import sys
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
import uuid

import math
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
import logging

from metrics.metrics import calculate_open_loop_metrics, calculate_trajectory_open_loop_metrics

from ibc import optimizers
from pilotnet import PilotNet, PilotNetConditional, PilotnetControl, PilotnetEBM
from efficient_net import effnetv2_s
from scripts.pt_to_onnx import convert_pt_to_onnx

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


def earth_mover_distance(input: Tensor, target: Tensor, square=False) -> Tensor:
    '''Adapted from: https://discuss.pytorch.org/t/implementation-of-squared-earth-movers-distance-loss-function-for-ordinal-scale/107927/2
    Change: option to take absolute instead of square, because the differences are very small.
    '''

    # convert to probability distribution
    input = F.softmax(input, dim=-1)
    target = F.softmax(target, dim=-1)
    diff_handler = torch.square if square else torch.abs

    return torch.mean(diff_handler(torch.cumsum(input, dim=-1) - torch.cumsum(target, dim=-1)))


class Trainer:

    def __init__(self, model_name=None, train_conf: train.TrainingConfig = None):  # todo:rename target_name->output_modality

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_name = train_conf.output_modality
        self.n_conditional_branches = train_conf.n_branches
        self.wandb_logging = False

        if train_conf.loss: # loss is not None, hence we're in training mode
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
                # MAE will be used in evaluation
                self.criterion = CrossEntropyLoss()
            else:
                print(f"Uknown loss function {train_conf.loss}")
                sys.exit()

            if train_conf.wandb_project:
                self.wandb_logging = True
                # wandb.init(project=wandb_project)

            if model_name:
                datetime_prefix = datetime.today().strftime('%Y%m%d%H%M%S')
                uuid_prefx = str(uuid.uuid4())[:8]
                self.save_dir = Path("models") / f"{datetime_prefix}_{uuid_prefx}_{model_name}"
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
            
            temp_reg_loss = None
            valid_temp_reg_loss = None

            progress_bar = tqdm(total=len(train_loader), smoothing=0)
            epoch_results = self.train_epoch(model, train_loader, optimizer, criterion, progress_bar, epoch)
            if not isinstance(epoch_results, tuple):
                train_loss = epoch_results
            elif len(epoch_results) == 2:
                train_loss, temp_reg_loss = epoch_results

            progress_bar.reset(total=len(valid_loader))
            epoch_results = self.evaluate(model, valid_loader, criterion, progress_bar, epoch, train_loss)
            if len(epoch_results) == 2:
                valid_loss, predictions = epoch_results
            elif len(epoch_results) == 3:
                valid_loss, predictions, valid_temp_reg_loss = epoch_results

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
            metrics['temporal_reg_loss'] = temp_reg_loss
            metrics['valid_temporal_reg_loss'] = valid_temp_reg_loss
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
        running_temporal_reg_loss = 0.0

        model.train()

        batch_wait_times = []

        ask_batch_timestamp = time.time()
        for i, (data, target_values, condition_mask) in enumerate(loader):
            recv_batch_timestap = time.time()
            batch_wait_time = recv_batch_timestap - ask_batch_timestamp
            batch_wait_times.append(batch_wait_time)
            logging.debug(f'\nModel waited for batch: {batch_wait_time * 1000:.2f}ms | avg: {np.mean(batch_wait_times) * 1000:.2f}ms (±{np.std(batch_wait_times) * 1000:.2f}) | max: {np.max(batch_wait_times) * 1000:.2f}ms | rate: {loader.batch_size / np.mean(batch_wait_times):.2f} FPS')

            optimizer.zero_grad()

            temporal_reg_loss = None
            batch_results = self.train_batch(model, data, target_values, condition_mask, criterion)
            if len(batch_results) == 2:
                predictions, loss = batch_results
            elif len(batch_results) == 3:
                predictions, loss, temporal_reg_loss = batch_results

            loss.backward()
            optimizer.step()
            if hasattr(self, 'scheduler'):
                self.scheduler.step()

            running_loss += loss.item()
            if temporal_reg_loss is not None:
                running_temporal_reg_loss += temporal_reg_loss

            progress_bar.update(1)
            pbar_description = f'epoch {epoch+1} | train loss: {(running_loss / (i + 1)):.4f} | temp reg loss: {running_temporal_reg_loss / (i+1):.4f}'
            if hasattr(self, 'reg_weight'):
                pbar_description += f' | reg weight: {self.reg_weight:.4f}'
            progress_bar.set_description(pbar_description)

            ask_batch_timestamp = time.time()

        avg_loss = running_loss / len(loader)
        if running_temporal_reg_loss > 0:
            avg_temporal_reg_loss = running_temporal_reg_loss / len(loader)
            return avg_loss, avg_temporal_reg_loss

        return avg_loss

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
                logging.debug(f'Model waited for batch: {(recv_batch_timestap - ask_batch_timestamp) * 1000:.2f} ms')

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

        self.model = PilotNet(train_conf.n_input_channels)
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


class EfficientNetTrainer(Trainer):

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


class EBMTrainer(Trainer):

    temporal_regularization_options = {
        'crossentropy': torch.nn.CrossEntropyLoss(),
        'l1': torch.nn.L1Loss(),
        'l2': torch.nn.MSELoss(),
        'emd': earth_mover_distance,
        'emd-squared': lambda a, b: earth_mover_distance(a, b, True)
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = PilotnetEBM()
        self.model.to(self.device)
        self.train_conf = kwargs['train_conf']

        optim_config, stochastic_optim_config = self._initialize_config(self.train_conf)

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

        inference_wrapper = optimizers.DFOptimizerConst if self.train_conf.use_constant_samples else optimizers.DFOptimizer

        self.temporal_regularization_criterion = self.temporal_regularization_options[self.train_conf.temporal_regularization_type]
        self.inference_model = inference_wrapper(self.model, stochastic_optim_config)
        self.inference_model.to(self.device)
        self.steps = 0
        self.reg_weight = self.train_conf.temporal_regularization

    def _initialize_config(self, train_conf):
        """Initialize train state based on config values."""

        optim_config = optimizers.OptimizerConfig(
            learning_rate=train_conf.learning_rate,
            weight_decay=train_conf.weight_decay,
        )

        target_bounds = torch.tensor([[-train_conf.steering_bound], [train_conf.steering_bound]]).to(self.device)
        stochastic_optim_config = optimizers.DerivativeFreeConfig(
            bounds=target_bounds,
            train_samples=train_conf.stochastic_optimizer_train_samples,
            inference_samples=train_conf.stochastic_optimizer_inference_samples,
            iters=train_conf.stochastic_optimizer_iters,
        )

        return optim_config, stochastic_optim_config

    def calc_temporal_regularization(self, logits: Tensor, eval=False) -> Tensor:
        """
        Calculate the temporal regularization loss for the given (unshuffled) logits and target indices.
        """

        if self.train_conf.temporal_regularization_ignore_target and not eval:
            # ignore (always changing) ground truth
            logits[:, 0] = 0.

        odd_samples = logits[::2, :]
        even_samples = logits[1::2, :]

        if odd_samples.shape != even_samples.shape: return 0

        return self.temporal_regularization_criterion(odd_samples, even_samples)

    def predict(self, _, dataloader):
        all_predictions = []
        inference_model = self.inference_model
        inference_model.eval()

        with torch.no_grad():
            progress_bar = tqdm(total=len(dataloader), smoothing=0)
            progress_bar.set_description("Model predictions")
            for i, (data, target_values, condition_mask) in enumerate(dataloader):
                inputs = data['image'].to(self.device)
                predictions, energy = inference_model(inputs)
                all_predictions.extend(predictions.cpu().squeeze().numpy())
                progress_bar.update(1)

        return np.array(all_predictions)

    # adopted from https://github.com/kevinzakka/ibc
    def train_batch(self, _, input, target, __, ___):
        inputs = input['image'].to(self.device)
        target = target.to(self.device, torch.float32)

        logging.debug(f'inputs: {inputs.shape} {inputs.dtype}')
        logging.debug(f'target: {target.shape} {target.dtype}')

        # Generate N negatives, one for each element in the batch: (B, N, D).
        negatives = self.inference_model.sample(inputs.size(0))

        # Merge target and negatives: (B, N+1, D).
        targets = torch.cat([target.unsqueeze(dim=1), negatives], dim=1)
        logging.debug(f'merged targets (should be [B, N+1, D]): {targets.shape} {targets.dtype}')

        # Generate a random permutation of the positives and negatives.
        permutation = torch.rand(targets.size(0), targets.size(1)).argsort(dim=1)
        logging.debug(f'permutation: {permutation.shape} {permutation.dtype}')
        targets = targets[torch.arange(targets.size(0)).unsqueeze(-1), permutation]
        logging.debug(f'permuted targets: {targets.shape} {targets.dtype}')

        # Get the original index of the positive. This will serve as the class label
        # for the loss.
        ground_truth = (permutation == 0).nonzero()[:, 1].to(self.device)

        # For every element in the mini-batch, there is 1 positive for which the EBM
        # should output a low energy value, and N negatives for which the EBM should
        # output high energy values.
        energy = self.model(inputs, targets)

        # Interpreting the energy as a negative logit, we can apply a cross entropy loss
        # to train the EBM.
        logits = -1.0 * energy

        loss = self.criterion(logits, ground_truth)
        temporal_regularization_loss = None

        if self.train_conf.temporal_regularization:
            # calculate weight
            reg_weight = self.train_conf.temporal_regularization

            if self.train_conf.temporal_regularization_schedule == 'exponential':
                k = self.train_conf.temporal_regularization_schedule_k
                reg_weight *= 1-math.exp(-1 * k * self.steps)
            elif self.train_conf.temporal_regularization_schedule == 'linear':
                n = self.train_conf.temporal_regularization_schedule_n
                reg_weight *= self.steps * n

            self.reg_weight = reg_weight
            logging.debug('temporal regularization weight: {}'.format(reg_weight))

            # calculcate loss
            logits_unshuffled = logits[torch.arange(logits.size(0)).unsqueeze(-1), torch.argsort(permutation)]
            temporal_regularization_loss = self.calc_temporal_regularization(logits_unshuffled)
            loss += reg_weight * temporal_regularization_loss

        self.optimizer.zero_grad(set_to_none=True)
        self.steps += 1

        if temporal_regularization_loss is not None:
            return energy, loss, temporal_regularization_loss.item()

        return energy, loss

    @torch.no_grad()
    def evaluate(self, _, iterator, __, progress_bar, epoch, train_loss):
        inference_model = self.inference_model
        inference_model.eval()
        all_predictions = []

        inference_times = []

        epoch_mae = 0.0
        epoch_temporal_reg_loss = 0.0
        ask_batch_timestamp = time.time()
        for i, (input, target, _) in enumerate(iterator):
            recv_batch_timestap = time.time()
            logging.debug(f'\nModel waited for batch: {(recv_batch_timestap - ask_batch_timestamp) * 1000:.2f} ms')

            inputs = input['image'].to(self.device)
            target = target.to(self.device, torch.float32)
            logging.debug(f'target.shape: {target.shape}, target.dtype: {target.dtype}, min: {target.min()}, max: {target.max()}')

            inference_start = time.time()
            preds, energy = inference_model(inputs)
            inference_end = time.time()

            inference_time = inference_end - inference_start
            inference_times.append(inference_time)

            logging.debug(f'inference time: {inference_time} | avg : {np.mean(inference_times)} | max: {np.max(inference_times)} | min: {np.min(inference_times)}')

            if self.train_conf.temporal_regularization:
                logits = -1 * energy
                temporal_regularization_loss = self.calc_temporal_regularization(logits, eval=True)
                epoch_temporal_reg_loss += temporal_regularization_loss.item()

            mae = F.l1_loss(preds, target.view(-1, 1))
            epoch_mae += mae.item()

            all_predictions.extend(preds.cpu().squeeze().numpy())

            progress_bar.update(1)
            progress_bar.set_description(f'epoch {epoch + 1} | train loss: {train_loss:.4f} | valid loss: {(epoch_mae / (i + 1)):.4f} | valid temporal reg loss: {(epoch_temporal_reg_loss / (i + 1)):.4f}')

            ask_batch_timestamp = time.time()

        avg_mae = epoch_mae / len(iterator)
        avg_temp_reg_loss = epoch_temporal_reg_loss / len(iterator)
        result = np.array(all_predictions)
        return avg_mae, result, avg_temp_reg_loss

    def save_onnx(self, _, __):
        pt_models = [f'{self.save_dir}/last.pt', f'{self.save_dir}/best.pt']

        for pt_model_path in pt_models:

            pure_model_args = ['--file', pt_model_path, '--output', pt_model_path.replace('.pt', '-pure.onnx'),
                               '--samples', str(self.inference_model.inference_samples), '--bs', '32',
                               '--steering-bound', str(self.inference_model.bounds.max().item())]

            dfo_model_args = ['--file', pt_model_path, '--output', pt_model_path.replace('.pt', '-dfo.onnx'),
                              '--with-dfo', '--iters', str(self.inference_model.iters),
                              '--samples', str(self.inference_model.inference_samples), '--bs', '1',
                               '--steering-bound', str(self.inference_model.bounds.max().item())]

            if self.train_conf.use_constant_samples:
                dfo_model_args.append('--use-constant-samples')
            
            pure_model_path = convert_pt_to_onnx(pure_model_args)
            dfo_model_path = convert_pt_to_onnx(dfo_model_args)

            if self.wandb_logging:
                wandb.save(pure_model_path)
                wandb.save(dfo_model_path)

