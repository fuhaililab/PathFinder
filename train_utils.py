"""
Training utils files
"""
import logging
import os
import queue
import shutil
import time
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import tqdm
import ujson as json
from sklearn.metrics import roc_curve, auc
from torch import Tensor, LongTensor


class LoadClassifierDataset(data.Dataset):
    r"""construct classification dataset with scRNA-seq expression data and pre-defined signaling network
    Args:
        expression (np.array): scRNA-seq expression data.
        gs_path (str): Path of hallmark gene set database.
        gene_list (list): Gene symbol list of input dataset.
        in_degree (Tensor): In degree for each genes.
        out_degree (Tensor): Out degree fpr each genes.
        shortest_path_length (LongTensor): The length of the shortest path between each gene pair.
        edge_types (LongTensor): Edge type for each edge in the graph.
        label (np.array): Label of each sample.
    """

    def __init__(self,
                 expression: np.array,
                 gs_path: str,
                 gene_list: list,
                 in_degree: Tensor,
                 out_degree: Tensor,
                 shortest_path_length: LongTensor,
                 edge_types: LongTensor,
                 label: np.array):
        super(LoadClassifierDataset, self).__init__()

        self.gene_list = gene_list
        self.expression = torch.from_numpy(expression).float()
        self.label = torch.from_numpy(label).long()
        self.num_nodes = len(gene_list)
        self.in_degree = in_degree
        self.out_degree = out_degree
        self.shortest_path_length = shortest_path_length
        self.edge_types = edge_types
        self.node_index = torch.tensor([i for i in range(len(self.gene_list))])

        # hallmark gene set feature
        with open(gs_path) as f:
            gene_feature_dict = json.load(f)
        gene_feature = np.zeros([len(gene_list), 50], dtype=np.int32)
        for i in range(len(gene_list)):
            gene = gene_list[i]
            gs_list = gene_feature_dict.get(gene.split("_")[0])
            if gs_list is not None and len(gs_list) > 0:
                for j in gs_list:
                    gene_feature[i, j] = 1
        self.gene_feature = torch.from_numpy(gene_feature).float()

    def __len__(self):
        return self.expression.size(0)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, LongTensor, LongTensor, LongTensor, LongTensor]:
        cell_expression = self.expression[idx]
        x = cell_expression.unsqueeze(-1)
        y = self.label[idx]
        return (x, y, self.in_degree, self.out_degree, self.node_index, self.edge_types)


def classifier_collate_fn(examples: list) -> Tuple[Tensor, Tensor, LongTensor, LongTensor, LongTensor, LongTensor, Tensor]:
    r"""Create batch tensors from a list of individual examples returned Merge examples of different length by padding
    all examples to the maximum length in the batch.
    Args:
        examples (list): List of tuples

    """

    def merge_features(tensor_list):
        tensor_list = [tensor.unsqueeze(0) for tensor in tensor_list]
        return torch.cat(tensor_list, dim=0)

    # Group by tensor type
    x_list, y_list, in_deg_list, out_deg_list, node_index_list, edge_type_list = zip(*examples)
    batch_x = merge_features(x_list)
    batch_y = torch.tensor(y_list)
    batch_in_deg = merge_features(in_deg_list)
    batch_out_deg = merge_features(out_deg_list)
    edge_types = merge_features(edge_type_list)
    batch_node_index = merge_features(node_index_list)
    batch_mask = torch.ones([len(x_list), x_list[0].size(0)])
    return (batch_x, batch_y, batch_in_deg, batch_out_deg, edge_types, batch_node_index, batch_mask)


class AverageMeter:
    r"""Keep track of average values over time.
    Adapted from:
        > https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        r"""Reset meter."""
        self.__init__()

    def update(self, val: float, num_samples: Optional[int] = 1):
        r"""Update meter with new value `val`, the average of `num` samples.

        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count


class EMA:
    r"""Exponential moving average of model parameters.
    Args:
        model (nn.Module): Model with parameters whose EMA will be kept.
        decay (float): Decay rate for exponential moving average.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model: nn.Module, num_updates: float):
        decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = \
                    (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model: nn.Module):
        r"""Assign exponential moving average of parameter values to the
        respective parameters.
        Args:
            model (nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model: nn.Module):
        r"""Restore original parameters to a model. That is, put back
        the values that were in each parameter at the last call to `assign`.
        Args:
            model (nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]


class CheckpointSaver:
    r"""Class to save and load model checkpoints.

    Save the best checkpoints as measured by a metric value passed into the
    `save` method. Overwrite checkpoints with better checkpoints once
    `max_checkpoints` have been saved.

    Args:
        save_dir (str): Directory to save checkpoints.
        max_checkpoints (int): Maximum number of checkpoints to keep before
            overwriting old ones.
        metric_name (str): Name of metric used to determine best model.
        maximize_metric (bool): If true, best checkpoint is that which maximizes
            the metric value passed in via `save`. Otherwise, best checkpoint
            minimizes the metric.
        log (logging.Logger): Optional logger for printing information.
    """

    def __init__(self,
                 save_dir: str,
                 max_checkpoints: int,
                 metric_name: str,
                 maximize_metric: bool = False,
                 log: logging.Logger = None):
        super(CheckpointSaver, self).__init__()

        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.log = log
        self._print(f"Saver will {'max' if maximize_metric else 'min'}imize {metric_name}...")

    def is_best(self, metric_val: float) -> bool:
        r"""Check whether `metric_val` is the best seen so far.

        Args:
            metric_val (float): Metric value to compare to prior checkpoints.
        """
        if metric_val is None:
            # No metric reported
            return False

        if self.best_val is None:
            # No checkpoint saved yet
            return True

        return ((self.maximize_metric and self.best_val < metric_val)
                or (not self.maximize_metric and self.best_val > metric_val))

    def _print(self, message: str):
        r"""Print a message if logging is enabled."""
        if self.log is not None:
            self.log.info(message)

    def save(self,
             step: int,
             model_dict: nn.Module,
             metric_val: float,
             device: str):
        r"""Save model parameters to disk.

        Args:
            step (int): Total number of examples seen during training so far.
            model (nn.Module): Model to save.
            metric_val (float): Determines whether checkpoint is best so far.
            device (str): Device where model resides.
        """

        checkpoint_path = os.path.join(self.save_dir, f'step_{step}')
        for name, model in model_dict.items():
            ckpt_dict = {
                'model_name': model.__class__.__name__,
                'model_state': model.cpu().state_dict(),
                'step': step
            }

            model.to(device)
            torch.save(ckpt_dict, f"{checkpoint_path}{name}.pth.tar")
        self._print(f'Saved checkpoint: {checkpoint_path}')

        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, 'best')
            for name in model_dict.keys():
                shutil.copy(f"{checkpoint_path}{name}.pth.tar", f"{best_path}{name}.pth.tar")

            self._print(f'New best checkpoint at step {step}...')

        # Add checkpoint path to priority queue (lowest priority removed first)
        if self.maximize_metric:
            priority_order = metric_val
        else:
            priority_order = -metric_val

        self.ckpt_paths.put((priority_order, checkpoint_path))

        # Remove a checkpoint if more than max_checkpoints have been saved
        if self.ckpt_paths.qsize() > self.max_checkpoints:
            _, worst_ckpt = self.ckpt_paths.get()
            try:
                for name in model_dict.keys():
                    os.remove(f"{worst_ckpt}{name}.pth.tar")
                self._print(f'Removed checkpoint: {worst_ckpt}')
            except OSError:
                # Avoid crashing if checkpoint has been removed or protected
                pass


def load_model(model: nn.Module,
               checkpoint_path: str,
               gpu_ids: list,
               return_step: bool = True) -> nn.Module:
    r"""Load model parameters from disk.

    Args:
        model (nn.Module): Load parameters into this model.
        checkpoint_path (str): Path to checkpoint to load.
        gpu_ids (list): GPU IDs for DataParallel.
        return_step (bool): Also return the step at which checkpoint was saved.

    Returns:
        model (nn.Module): Model loaded from checkpoint.
        step (int): Step at which checkpoint was saved. Only if `return_step`.
    """
    device = f"cuda:{gpu_ids[0]}" if gpu_ids else 'cpu'
    ckpt_dict = torch.load(checkpoint_path, map_location=device)

    # Build model, load parameters
    model.load_state_dict(ckpt_dict['model_state'])

    if return_step:
        step = ckpt_dict['step']
        return model, step

    return model


def get_available_devices() -> Tuple[str, list]:
    r"""Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids


def get_save_dir(base_dir: str, name: str, type: str, id_max: int = 100) -> str:
    r"""Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).

    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        type (str): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.

    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        subdir = type
        save_dir = os.path.join(base_dir, subdir, f'{name}-{uid:02d}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError('Too many save directories created with the same name. \
                       Delete old save directories or use another name.')


def get_logger(log_dir: str, name: str) -> logging.Logger:
    r"""Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.

    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.

    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """

    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.

        See Also:
            > https://stackoverflow.com/questions/38543506
        """

        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class MetricsMeter:
    r"""Keep track of model performance.
    """

    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        self.threshold = 0.5
        self.prediction = np.array([1])
        self.label = np.array([1])

    def reset(self):
        """Reset meter."""
        self.__init__()

    def update(self, input: Tensor, target: Tensor):
        """Update meter with new result

        Args:
            input (torch.tensor, Batch_size*1): predicted probability tensor.
            target (torch.tensor, Batch_size*1): ground true, 1 represent positive

        """
        predict = (input > self.threshold).int()
        self.TP += (target[torch.where(predict == 1)] == 1).sum().item()
        self.FP += (target[torch.where(predict == 1)] == 0).sum().item()
        self.TN += (target[torch.where(predict == 0)] == 0).sum().item()
        self.FN += (target[torch.where(predict == 0)] == 1).sum().item()
        input = input.view(-1).numpy()
        target = target.view(-1).numpy()
        self.prediction = np.concatenate([self.prediction, input], axis=-1)
        self.label = np.concatenate([self.label, target], axis=-1)

    def return_metrics(self) -> dict:
        recall = self.TP / (self.TP + self.FN + 1e-30)
        precision = self.TP / (self.TP + self.FP + 1e-30)
        specificity = self.TN / (self.TN + self.FP + 1e-30)
        accuracy = (self.TP + self.TN) / (self.TP + self.FP + self.TN + self.FN + 1e-30)
        F1 = self.TP / (self.TP + 0.5 * (self.FP + self.FN) + 1e-30)
        fpr, tpr, thresholds = roc_curve(self.label[1:], self.prediction[1:])
        AUC = auc(fpr, tpr)
        metrics_result = {'Accuracy': accuracy,
                          "Recall": recall,
                          "Precision": precision,
                          "Specificity": specificity,
                          "F1": F1,
                          "AUC": AUC,
                          "fpr": fpr,
                          "tpr": tpr,
                          "thresholds": thresholds
                          }
        return metrics_result

def get_seed(seed=234) -> int:
    r"""Return random seed based on current time.
    Args:
        seed (int): base seed.
    """
    t = int(time.time() * 1000.0)
    seed = seed + ((t & 0xff000000) >> 24) + ((t & 0x00ff0000) >> 8) + ((t & 0x0000ff00) << 8) + ((t & 0x000000ff) << 24)
    return seed

def split_train_val_test(index: list,
                         seed: int,
                         val_ratio: float,
                         test_ratio: float,
                         label: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_index, val_index = train_test_split(index, test_size=(test_ratio + val_ratio), shuffle=True,
                                              random_state=seed, stratify=label)
    val_index, test_index = train_test_split(val_index, test_size=(test_ratio / (test_ratio + val_ratio)),
                                             shuffle=True, random_state=seed, stratify=label[val_index])

    return train_index, val_index, test_index
