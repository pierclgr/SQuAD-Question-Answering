import os
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from src.dataset import SQuADJSONDataset
from src.config import SQuADModelConfig
import builtins
import torch.nn as nn
import argparse

if 'COLAB_TPU_ADDR' in os.environ:
    import torch_xla.core.xla_model as xm

from src.experiment import Experiment


# define main method
def main() -> None:
    """
    Main method of the script
    """

    print = builtins.print

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("train", help="Path of the training set json file")
    parser.add_argument("valid", help="Path of the validation set json file")
    parser.add_argument("-tst", "--test", help="Path of the tesing set json file", required=False)
    parser.add_argument("config", help="Path of the configuration json file")
    args = parser.parse_args()

    # if runtime has a TPU, use it
    if 'COLAB_TPU_ADDR' in os.environ:
        # define a flag that specifies if the TPU is going to be used
        use_tpu = True
    else:
        use_tpu = False

    # configuring device
    if use_tpu:
        device = xm.xla_device()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # configure print function
    if use_tpu:
        print = xm.master_print

    print(f"\n>>> Using {device} device")

    # extract the specified configuration json file path
    config_path = args.config

    # load the configuration file
    print("\n>>> Loading configuration...")
    config = SQuADModelConfig(config_path)
    print(config)

    # extract the specified splits json file paths
    train_json_path = args.train
    val_json_path = args.valid
    test_json_path = args.test

    # create the SQuADJSONDataset object and dataframe for the train split
    print("\n>>> Loading the training split...")

    train_json = SQuADJSONDataset.from_path(train_json_path)
    train_df = train_json.to_dataframe()

    # create the SQuADJSONDataset object and dataframe for the validation split
    print(">>> Loading the validation split...")

    val_json = SQuADJSONDataset.from_path(val_json_path)
    val_df = val_json.to_dataframe()

    # if available, create the SQuADJSONDataset object and dataframe for the testing split
    if test_json_path:
        print(">>> Loading the testing split...")

        test_json = SQuADJSONDataset.from_path(test_json_path)
        test_df = test_json.to_dataframe()
    else:
        test_df = None

    # define experimentation arguments
    print("\n>>> Configuring experimentation...")
    training_kwargs = dict(batch_size=config.batch_size, device=device)
    dataset_kwargs = dict(train_df=train_df, val_df=val_df, test_df=test_df)
    model_kwargs = dict(model_name=config.pretrained_model,
                        type=config.type,
                        model_params=dict(dropout_rate=config.dropout_rate))

    # create experiment object
    experiment_kwargs = {**training_kwargs, **dataset_kwargs, **model_kwargs}
    if config.save_experiment:
        if config.type == "enhanced1":
            save_model_name = config.pretrained_model + "-enhanced1-model.pytorch"
        elif config.type == "enhanced2":
            save_model_name = config.pretrained_model + "-enhanced2-model.pytorch"
        else:
            save_model_name = config.pretrained_model + "-baseline-model.pytorch"
    else:
        save_model_name = None

    if config.type == "enhanced1":
        experiment_name = config.pretrained_model + "-enhanced1-experiment"
    elif config.type == "enhanced2":
        experiment_name = config.pretrained_model + "-enhanced2-experiment"
    else:
        experiment_name = config.pretrained_model + "-baseline-experiment"
    experiment = Experiment(experiment=experiment_name,
                            save_model_name=save_model_name, **experiment_kwargs)

    # create the optimizer
    optimizer_args = dict(lr=config.learning_rate)
    experiment.init_optimizer(AdamW, **optimizer_args)

    # create the learning rate scheduler
    scheduler_args = dict(num_warmup_steps=0, num_training_steps=len(experiment.train_dataloader) * config.num_epochs)
    experiment.init_lr_scheduler(get_linear_schedule_with_warmup, **scheduler_args)

    # create criterion
    criterion = nn.CrossEntropyLoss(ignore_index=experiment.train_dataloader.dataset.max_length).to(experiment.device)

    # run experiment
    experiment.fit(config.num_epochs, criterion, verbose=config.verbose)


if __name__ == "__main__":
    main()
