# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing
import warnings

from datasets.base import DataLoader
import datasets.registry
from foundations import hparams
from foundations import paths
from foundations.step import Step
from models.base import Model, DataParallel, DistributedDataParallel
import models.registry
from platforms.platform import get_platform
from training.checkpointing import restore_checkpoint
from training import optimizers
from training import standard_callbacks
from training.metric_logger import MetricLogger
import numpy as np
from tqdm import tqdm
from torch import autograd
import torch

try:
    import apex
    NO_APEX = False
except ImportError:
    NO_APEX = True


def gradient_normalize(arr):
    return(arr/np.linalg.norm(arr))


def train(
    training_hparams: hparams.TrainingHparams,
    model: Model,
    train_loader: DataLoader,
    output_location: str,
    callbacks: typing.List[typing.Callable] = [],
    start_step: Step = None,
    end_step: Step = None,
    collect_gradients: bool = True,
):

    """The main training loop for this framework.

    Args:
      * training_hparams: The training hyperparameters whose schema is specified in hparams.py.
      * model: The model to train. Must be a models.base.Model
      * train_loader: The training data. Must be a datasets.base.DataLoader
      * output_location: The string path where all outputs should be stored.
      * callbacks: A list of functions that are called before each training step and once more
        after the last training step. Each function takes five arguments: the current step,
        the output location, the model, the optimizer, and the logger.
        Callbacks are used for running the test set, saving the logger, saving the state of the
        model, etc. The provide hooks into the training loop for customization so that the
        training loop itself can remain simple.
      * start_step: The step at which the training data and learning rate schedule should begin.
        Defaults to step 0.
      * end_step: The step at which training should cease. Otherwise, training will go for the
        full `training_hparams.training_steps` steps.
      * collect_gradients: Whether to run an epoch at the end to collect the gradients.
    """

    # Create the output location if it doesn't already exist.
    if not get_platform().exists(output_location) and get_platform().is_primary_process:
        get_platform().makedirs(output_location)

    # Get the optimizer and learning rate schedule.
    model.to(get_platform().torch_device)
    optimizer = optimizers.get_optimizer(training_hparams, model)
    step_optimizer = optimizer
    lr_schedule = optimizers.get_lr_schedule(training_hparams, optimizer, train_loader.iterations_per_epoch)
    # ###########################################

    # Adapt for FP16.
    if training_hparams.apex_fp16:
        if NO_APEX: raise ImportError('Must install nvidia apex to use this model.')
        model, step_optimizer = apex.amp.initialize(model, optimizer, loss_scale='dynamic', verbosity=0)

    # Handle parallelism if applicable.
    if get_platform().is_distributed:
        model = DistributedDataParallel(model, device_ids=[get_platform().rank])
    elif get_platform().is_parallel:
        model = DataParallel(model)

    # Get the random seed for the data order.
    data_order_seed = training_hparams.data_order_seed

    # Restore the model from a saved checkpoint if the checkpoint exists.
    cp_step, cp_logger = restore_checkpoint(output_location, model, optimizer, train_loader.iterations_per_epoch)
    start_step = cp_step or start_step or Step.zero(train_loader.iterations_per_epoch)
    logger = cp_logger or MetricLogger()
    with warnings.catch_warnings():  # Filter unnecessary warning.
        warnings.filterwarnings("ignore", category=UserWarning)
        for _ in range(start_step.iteration): lr_schedule.step()

    # Determine when to end training.
    end_step = end_step or Step.from_str(training_hparams.training_steps, train_loader.iterations_per_epoch)
    if end_step <= start_step: return


    # The training loop.
    for ep in range(start_step.ep, end_step.ep + 1):

        # Ensure the data order is different for each epoch.
        train_loader.shuffle(None if data_order_seed is None else (data_order_seed + ep))

        for it, (examples, labels) in enumerate(train_loader):

            # Advance the data loader until the start epoch and iteration.
            if ep == start_step.ep and it < start_step.it: continue



            # Exit at the end step.
            if ep == end_step.ep and it == end_step.it: 
                # collecting the gradients at the end step
                if collect_gradients:
                    layer_names_grads = []
                    layers_to_compute_gradients_on = []
                    for name, tensor in model.named_parameters():
                        if tensor.requires_grad:
                            layer_names_grads += [name]
                            layers_to_compute_gradients_on += [tensor]

                    gradient_sum = [np.zeros(l.shape) for l in layers_to_compute_gradients_on]

                    # print([l.shape for l in layers_to_compute_gradients_on])
                    for examples, labels in tqdm(train_loader, 'Collecting gradients'):
                        examples = examples.to(device=get_platform().torch_device)
                        labels = labels.to(device=get_platform().torch_device)
                        predicted = model(examples)
                        loss = model.loss_criterion(predicted, labels)
                        gradients = autograd.grad(loss, layers_to_compute_gradients_on, retain_graph=True) # needs retain graph because gradient information gets destroyed after first access
                        for i, g in enumerate(gradients):
                            gradient_sum[i] += np.abs(g.detach().cpu().numpy())
                    
                    for name, grad in zip(layer_names_grads, gradient_sum):
                        model.grads[name] = grad
                    # print(f'names in train py: {model.grads.keys()}')
                    

            # Run the callbacks.
            step = Step.from_epoch(ep, it, train_loader.iterations_per_epoch)
            for callback in callbacks: callback(output_location, step, model, optimizer, logger)

            if ep == end_step.ep and it == end_step.it: 
                return

            # Otherwise, train.
            examples = examples.to(device=get_platform().torch_device)
            labels = labels.to(device=get_platform().torch_device)

            step_optimizer.zero_grad()
            model.train()
            loss = model.loss_criterion(model(examples), labels)
            

            if training_hparams.apex_fp16:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            # if we are on text, we need to do gradient clipping. The strength of the clipping is hardcoded for AGNews CharCNN right now.
            if isinstance(train_loader, datasets.base.TextLoader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 400)
            
            # Step forward. Ignore extraneous warnings that the lr_schedule generates.
            step_optimizer.step()
            with warnings.catch_warnings():  # Filter unnecessary warning.
                warnings.filterwarnings("ignore", category=UserWarning)
                lr_schedule.step()

        
    get_platform().barrier()


def standard_train(
  model: Model,
  output_location: str,
  dataset_hparams: hparams.DatasetHparams,
  training_hparams: hparams.TrainingHparams,
  start_step: Step = None,
  verbose: bool = True,
  evaluate_every_epoch: bool = True
):
    """Train using the standard callbacks according to the provided hparams."""

    # If the model file for the end of training already exists in this location, do not train.
    iterations_per_epoch = datasets.registry.iterations_per_epoch(dataset_hparams)
    train_end_step = Step.from_str(training_hparams.training_steps, iterations_per_epoch)
    if (models.registry.exists(output_location, train_end_step) and
        get_platform().exists(paths.logger(output_location))): return

    train_loader = datasets.registry.get(dataset_hparams, train=True)
    test_loader = datasets.registry.get(dataset_hparams, train=False)
    callbacks = standard_callbacks.standard_callbacks(
        training_hparams, train_loader, test_loader, start_step=start_step,
        verbose=verbose, evaluate_every_epoch=evaluate_every_epoch)
    train(training_hparams, model, train_loader, output_location, callbacks, start_step=start_step)
