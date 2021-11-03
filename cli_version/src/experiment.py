import math
import pathlib
import shutil
import time
import warnings
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, PreTrainedTokenizer
from src.dataset import SQuADDataset
from src.model import BaselineSQuADModel, EnhancedSQuADModel1, EnhancedSQuADModel2, LoadableModule
from src.train_loop import train
from src.validation_loop import validation


class Experiment:
    """
    Models a single training/validation example
    """

    _root_dir: pathlib.Path = pathlib.Path("experiments")

    @staticmethod
    def get_dataloader(dataset: SQuADDataset, batch_size: int, shuffle: bool) -> data.DataLoader:
        """
        Creates DataLoader from a given PyTorch Dataset

        Parameters
        ----------
        dataset: SQuADDataset
            PyTorch SQuAD dataset to create dataloader from
        batch_size: int
            size of the batches
        shuffle: bool
            flag to control the shuffling of the data in the dataset

        Returns
        -------
        data.DataLoader
            dataloader created from the input dataset
        """

        # set default dataloader parameters
        dataloder_default_params = dict(collate_fn=dataset.collate_fn,
                                        num_workers=2,
                                        pin_memory=True,
                                        )

        # create and return the dataloader
        return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **dataloder_default_params)

    @staticmethod
    def get_model_from_string(model_name: str, type: str, **kwargs) -> LoadableModule:
        """
        Creates the specified model given a pre-trained model string

        Parameters
        ----------
        model_name: str
            name of the pretrained model to use
        type: str
            define type (baseline or not) of the model
        **kwargs
            arguments of the model constructor

        Returns
        ------------------
        LoadableModule
            specified model with the given pretrained model
        """

        # define model based on the type property
        if type == "enhanced1":
            return EnhancedSQuADModel1(model_name=model_name, **kwargs)
        elif type == "enhanced2":
            return EnhancedSQuADModel2(model_name=model_name, **kwargs)
        else:
            return BaselineSQuADModel(model_name=model_name, **kwargs)

    @staticmethod
    def get_tokenizer_from_string(model_name: str) -> PreTrainedTokenizer:
        """
        Creates pretrained tokenizer given a string

        Parameters
        ----------
        model_name: str
            name of the pretrained tokenizer to use

        Returns
        -------
        PreTrainedTokenizer
            pretrained tokenizer
        """

        return AutoTokenizer.from_pretrained(model_name)

    def __init__(self,
                 experiment: str,
                 save_model_name: str = None,
                 **kwargs) -> None:
        """
        Constructor of the class

        Parameters
        ----------
        experiment: str
            name of the experiment
        save_model_name: str
            name of the model to save (default None)
        **kwargs
            arguments of the experimentation
        """

        # create experiment folder
        experiment_path = Experiment._root_dir / experiment
        if not experiment_path.exists():
            experiment_path.mkdir(parents=True)
        self.experiment_path = experiment_path

        # save model name and save model flag
        self.save_model = True if save_model_name else False
        if self.save_model:
            self.save_model_path = self.experiment_path / save_model_name

        # create tensorboard writer        
        self.log_writer = SummaryWriter(log_dir=str(experiment_path))

        # get device
        self.device = kwargs.pop("device")

        # get model
        type = kwargs.pop("type")  # model is baseline or not
        model_name = kwargs.pop("model_name")
        model_params = kwargs.pop("model_params", dict())  # allow default paramters
        assert isinstance(model_params, dict)
        self.model = Experiment.get_model_from_string(model_name, type, **model_params).to(self.device)

        # get tokenizer
        self.tokenizer = Experiment.get_tokenizer_from_string(model_name)

        # === DATALOADER ===
        # get dataframes
        train_df = kwargs.pop("train_df")
        val_df = kwargs.pop("val_df")

        # if available, get test dataframe
        test_df = kwargs.pop("test_df", None)

        # get datasets
        train_dataset = SQuADDataset(train_df, self.tokenizer)
        val_dataset = SQuADDataset(val_df, self.tokenizer)

        # create dataloaders
        batch_size = kwargs.pop("batch_size")
        self.train_dataloader = Experiment.get_dataloader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        self.val_dataloader = Experiment.get_dataloader(dataset=val_dataset, batch_size=batch_size * 2, shuffle=False)

        # if test set is available, get the test dataset and create the test dataloader
        self.testing = False
        if test_df is not None:
            self.testing = True
            test_dataset = SQuADDataset(test_df, self.tokenizer)
            self.test_dataloader = Experiment.get_dataloader(dataset=test_dataset, batch_size=batch_size * 2,
                                                             shuffle=False)

        # === Empty objects init later ===
        self.optimizer = None
        self.lr_scheduler = None

    def init_optimizer(self, optimizer_cls, **optimizer_kwargs) -> None:
        """
        Initialize the optimizer

        Parameters
        ----------
        optimizer_cls
            optimizer to use
        **optimizer_kwargs
            arguments of the optimizer
        """

        kwargs = {
            "params": self.model.parameters(),
            **optimizer_kwargs
        }

        self.optimizer = optimizer_cls(**kwargs)

    def init_lr_scheduler(self, lr_scheduler_cls, **lr_scheduler_kwargs) -> None:
        """
        Initialize learning rate scheduler

        Parameters
        ----------
        lr_scheduler_cls
            learning rate scheduler to use
        **lr_scheduler_kwargs
            arguments of the learning rate scheduler
        """

        kwargs = {
            "optimizer": self.optimizer,
            **lr_scheduler_kwargs
        }

        self.lr_scheduler = lr_scheduler_cls(**kwargs)

    def zip_experiment(self) -> None:
        """
        Zip the experiment folder locally to make it available for download
        """

        print(">>> Creating zip...")
        shutil.make_archive(base_name=str(self.experiment_path), format='zip', root_dir=str(self.experiment_path))
        save_path = self.experiment_path.parent / f"{self.experiment_path.name}.zip"
        print(f">>> Experiment zipped in {save_path}")

    def fit(self,
            epochs: int,
            criterion,
            verbose: bool = False,
            use_tpu: bool = False) -> None:
        """
        Runs the experiment

        Parameters
        ----------
        epochs: int
            number of total training epochs
        criterion
            loss function
        verbose: bool
            flag to control verbose mode (default False)
        use_tpu: bool
            flag to control the use of TPU (default False)
        """

        start_fit = time.time()

        assert self.optimizer, "Run 'init_optimizer'"

        if not self.lr_scheduler:
            warnings.warn("lr_scheduler is not initialized. You can ignore this message if it is intended.")

        for epoch in range(epochs):

            print(f"\n>>> Epoch {epoch + 1}...")

            start_time = time.time()

            print(f">>> Training...")
            train_status = train(self.log_writer, self.model, self.train_dataloader, criterion, self.optimizer,
                                 self.lr_scheduler, self.device, epoch, verbose, use_tpu)

            print(f"\n>>> Validating...")
            val_status = validation(self.log_writer, self.model, self.val_dataloader, criterion, self.device, epoch)

            end_time = time.time()

            print(f"\n>>> Epoch {epoch + 1}/{epochs} <<< Elapsed time: {math.floor(end_time - start_time)} s")
            print(">>> Train: ", train_status)
            print(">>> Val: ", val_status)

            self.log_writer.add_scalars("loss",
                                        {"train": train_status.loss.mean(as_float=True),
                                         "val": val_status.loss.mean(as_float=True)},
                                        global_step=epoch + 1)

            for train_metric, val_metric in zip(train_status.metrics, val_status.metrics):
                name = train_metric.name if hasattr(train_metric, "name") else train_metric.__class__.__name__
                self.log_writer.add_scalars(f"epoch/{name}",
                                            {"train": train_metric.mean(),
                                             "val": val_metric.mean()},
                                            global_step=epoch + 1)

        elapsed_time = time.time() - start_fit

        print(f"\n>>> Training completed in {math.ceil(elapsed_time)} s")

        if self.testing:
            print(f"\n>>> Testing...")
            test_status = validation(self.log_writer, self.model, self.test_dataloader, criterion, self.device, epochs)
            print(">>> Test: ", test_status)

        self.log_writer.close()

        if self.save_model:
            print(f"\n>>> Saving model in {self.save_model_path}")
            self.model.save_model(str(self.save_model_path), use_tpu=use_tpu)
            self.zip_experiment()
