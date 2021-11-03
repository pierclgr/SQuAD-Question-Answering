from abc import ABC, abstractmethod
from typing import List
import torch
import evaluate


# create a class to manage the metrics for the experiments
class Metric(ABC):
    """
    Class which manages the metrics used in the experimentations
    """

    def __init__(self) -> None:
        """
        Constructor method of the class
        """

        self.values = []

    @abstractmethod
    def __call__(self, y_true, y_pred) -> None:
        """
        Call method of the class

        Parameters
        ----------
        y_true
            ground truth
        y_pred
            predictions
        """

        raise NotImplemented

    def update(self, y_true, y_pred) -> None:
        """
        Update the metric history with the new data

        Parameters
        ----------
        y_true
            ground truth
        y_pred
            predictions
        """

        values = self(y_true, y_pred)
        self.values.extend(values)

    def mean(self) -> float:
        """
        Computes mean of the metrics

        Returns
        -------
        float
            average of the metric
        """

        return sum(self.values) / len(self.values)

    def sum(self) -> float:
        """
        Computes sum of the metrics history

        Returns
        -------
        float
            sum of the metric
        """

        return sum(self.values)

    def percentage(self) -> float:
        """
        Converts the metric to percentage measure

        Returns
        -------
        float
            mean of the metric in percentage
        """

        return self.mean() * 100

    def __str__(self) -> str:
        """
        Computes a string representation of the current object

        Returns
        -------
        str
            string representation of the metrics
        """

        value = f"{self.mean():.2f} ( {self.percentage():.2f}% )"

        if hasattr(self, "name"):
            return f"{self.name}={value}"

        return f"{self.__class__.__name__}={value}"


# create a class to manage the loss metric
class Loss:
    """
    Class that manages the loss metric
    """

    def __init__(self) -> None:
        """
        Constructor of the class
        """

        self.values = []

    def update(self, value) -> None:
        """
        Update the loss history

        Parameters
        ----------
        value
            loss value to add to the history
        """

        self.values.append(value)

    def mean(self, as_float=False):
        """
        Compute mean of the loss history

        Parameters
        ----------
        as_float: bool
            flag that defines if the mean must be returned as float
        """

        if isinstance(self.values[0], torch.Tensor):
            values = torch.stack(self.values, dim=0).mean()
            return values.item() if as_float else values
        else:
            return sum(self.values) / len(self.values)

    def __str__(self) -> str:
        """
        Computes a string representation of the loss metric

        Returns
        -------
        str
            string representation of the loss metric
        """

        return f"loss={self.mean(as_float=True):.2f}"


# create a class to manage the accuracy metric
class AccuracyMetric(Metric):
    """
    Class to manage the accuracy metric
    """

    # define global attribute name
    name = "accuracy"

    def __call__(self, y_true, y_pred) -> list:
        """
        Compute accuracy on the batch per-sample

        Parameters
        ----------
        y_true
            ground truths
        y_pred
            predictions

        Returns
        -------
        list
            accuracy on the batch per sample
        """

        return [evaluate.compute_exact(a_gold, a_pred) for a_gold, a_pred in zip(y_true, y_pred)]


# create a class to manage the F1 metric
class F1Metric(Metric):
    """
    Class to manage the F1 metric
    """

    # define global attribute name
    name = "f1"

    def __call__(self, y_true, y_pred) -> list:
        """
        Compute f1 on the batch per-sample

        Parameters
        ----------
        y_true
            ground truths
        y_pred
            predictions

        Returns
        -------
        list
            f1 on the batch per sample
        """

        return [evaluate.compute_f1(a_gold, a_pred) for a_gold, a_pred in zip(y_true, y_pred)]


# create a class which shows the status of the loss and metrics
class Status:
    """
    Shows the status of the loss and metrics
    """
    def __init__(self, loss: Loss, metrics: List[Metric]) -> None:
        """
        Constructor of the class

        Parameters
        ----------
        loss: Loss
            loss to memorize
        metrics: List[Metric]
            metric history to memorize
        """

        self.loss = loss
        self.metrics = metrics

    def __str__(self) -> str:
        """
        Computes a string representation of the loss and metric status

        Returns
        -------
        str
            string representation of the loss and metric status
        """

        metrics_str = " ".join([str(metric) for metric in self.metrics])
        return f"{self.loss} {metrics_str}"


def get_factor_loss_optimizer(criterion, batch_size: int) -> int:
    """
    Compute the factor that multiplies the loss

    Parameters
    ----------
    criterion
        loss function
    batch_size: int
        size of the batches
    """

    if hasattr(criterion, "reduction"):
        reduction = getattr(criterion, "reduction")
        if reduction == "mean":
            return batch_size
        elif reduction == "sum":
            return 1
        else:
            raise RuntimeError(f"Unknow reduction: {reduction}. Expected: (mean, sum).")
    else:
        raise RuntimeError(f"No reduction attribute in {criterion}.")
