import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import os

from src.utilities import input_ids_to_list_strs, dict_to_device

if 'COLAB_TPU_ADDR' in os.environ:
    import torch_xla.core.xla_model as xm

from src.metrics import Loss, AccuracyMetric, F1Metric, get_factor_loss_optimizer, Status


# define the function that performs a single training epoch
def train(log_writer: SummaryWriter,
          model: nn.Module,
          train_loader: data.DataLoader,
          criterion,
          optimizer,
          scheduler,
          device,
          current_epoch: int,
          verbose: bool = False,
          use_tpu: bool = False):
    """
    Performs a single training epoch

    Parameters
    ----------
    log_writer: SummaryWriter
        tensorboard SummaryWriter of the experiment
    model: nn.Module
        model that is being trained
    train_loader: data.DataLoader
        dataloader of the training set
    criterion
        loss function
    optimizer
        optimizer parameters
    device
        PyTorch device on which the network is stored
    current_epoch:int
        current training epoch
    verbose: bool
        flag to control verbose mode (default False)
    use_tpu: bool
        flag that defines if the TPU is being used or not (default False)
    """

    # Set the model in training mode: this change the behaiour of Dropout, BatchNorm etc.
    model.train()

    # define the loss and metrics histories as empty lists
    total_loss = Loss()
    total_metrics = [AccuracyMetric(), F1Metric()]

    # ignore all the start and end labels that are over the maximum length of the model
    ignored_index = train_loader.dataset.max_length

    # for each sample in the training set
    for i, (_, tokenized_examples, contexts, _, start_positions, end_positions, _) in enumerate(
            tqdm(train_loader, total=len(train_loader), unit='batch', leave=True, position=0)):

        # zero out any previously calculated gradients
        optimizer.zero_grad()

        # extract batch size from the batches
        batch_size = tokenized_examples['input_ids'].size(0)

        # global step
        batch_index = i + 1
        global_step = current_epoch * batch_size + batch_index

        # pop out the offset mapping from the tokenized_examples
        offset_mappings = tokenized_examples.pop("offset_mapping")

        # move tensors to device
        start_positions = start_positions.to(device)
        end_positions = end_positions.to(device)
        tokenized_examples = dict_to_device(tokenized_examples, device)

        # compute start/end logits using the model
        start_logits, end_logits = model(tokenized_examples)

        # clamp the start and end labels in order to remove the ones outside the sequence maximum length
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)

        # compute the start and end losses and average them to get the full loss
        loss_start = criterion(start_logits, start_positions)
        loss_end = criterion(end_logits, end_positions)
        loss = (loss_start + loss_end) / 2

        # backprop loss
        loss.backward()

        # optimizer step
        if use_tpu:
            # optimizer_step also uses reduce_gradient which is useful for gradient clipping
            xm.optimizer_step(optimizer, barrier=True)
            xm.mark_step()
        else:
            # clip gradients to avoid exploding gradient problem
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # scheduler step
        if scheduler:
            scheduler.step()

        # update using tensor in order to avoid computation on tpu
        total_loss.update(loss.item() * get_factor_loss_optimizer(criterion, batch_size))

        # compute start/end probabilities and the indexes of the start and end tokens from it
        start_indexes = F.log_softmax(start_logits, dim=-1).detach().cpu()
        start_indexes = start_indexes.argmax(dim=-1)
        end_indexes = F.log_softmax(end_logits, dim=-1).detach().cpu()
        end_indexes = end_indexes.argmax(dim=-1)

        # extract the real answer from the original context (ground truths)
        ground_truths = input_ids_to_list_strs(contexts, offset_mappings, start_positions, end_positions)

        # extract the predicted answer from the original context (predictions)
        predictions = input_ids_to_list_strs(contexts, offset_mappings, start_indexes, end_indexes)

        # compute metrics scores
        for metric in total_metrics:
            metric.update(ground_truths, predictions)

        # report on tensorboard
        log_writer.add_text("train/answers/ground_truth", ground_truths[0], global_step)
        log_writer.add_text("train/answers/prediction", predictions[0], global_step)

        # if verbose, print the metrics each 20 batch
        if verbose:
            if batch_index % 100 == 0:
                print(f"\n>>> Batch {batch_index}")
                print(f">>> Train: {Status(total_loss, total_metrics)}")

    return Status(total_loss, total_metrics)
