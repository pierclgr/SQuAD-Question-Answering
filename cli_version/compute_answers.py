import builtins
import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import json
import sys
import argparse
from itertools import groupby

from src.utilities import dict_to_device, input_ids_to_list_strs
from src.model import LoadableModule
from src.experiment import Experiment
from src.dataset import SQuADJSONDataset, SQuADDataset

if 'COLAB_TPU_ADDR' in os.environ:
    import torch_xla.core.xla_model as xm


# define a method to extract the answer
def group_predictions_by_question_id(idxs: list, predictions: list) -> dict:
    """
    Groups answers by id and cleans question that have multiple answers (because of the splitting when the feature is
    longer than the max length of the model) and remove multiple answers to keep just one

    Parameters
    ----------
    idxs: list
        list of indices
    predictions: list
        list of predictions

    Returns
    -------
    dict
        dictionary containing for each question a single answer
    """

    def get_answer(groups: list) -> str:
        """
        Extract the answer from the group of answers

        Parameters
        ----------
        groups: list
            list of multiple answers for a question

        Returns
        -------
        str
            single answer to the question
        """

        # remove empty answer
        answers = [answer for _, answer in groups if answer != ""]

        if not answers:
            # no answer predicted
            return ""
        else:
            return answers[0]

    keyfunc = lambda elem: elem[0]
    idx_and_predictions = sorted(zip(idxs, predictions), key=keyfunc)
    idx_and_predictions = groupby(idx_and_predictions, keyfunc)
    idx_and_predictions = {key: get_answer(groups) for key, groups in idx_and_predictions}

    return idx_and_predictions


# define main method
def main() -> None:
    """
    Main method of the script
    """

    print = builtins.print

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

    # extract the specified test split json file path
    parser = argparse.ArgumentParser()
    parser.add_argument("test", help="Path of the testing set json file")
    args = parser.parse_args()
    test_json_path = args.test

    # create the SQuADJSONDataset objects for the test split
    print("\n>>> Loading the test split...")

    test_json = SQuADJSONDataset.from_path(test_json_path)

    # create pandas DataFrames of the test split
    print("\n>>> Creating DataFrame...")

    test_df = test_json.to_dataframe()

    # create and load the trained model
    print("\n>>> Loading trained model...")
    model = LoadableModule.load_model(sys.path[0] + "/models/" + "final-model.pytorch", device, use_tpu).to(device)
    print(f">>> Using model {type(model).__name__} with {model.model_name} pre-trained")

    # create huggingface tokenizer
    print("\n>>> Loading tokenizer...")
    tokenizer = Experiment.get_tokenizer_from_string(model.model_name)

    # Create Pytorch dataset
    test_dataset = SQuADDataset(test_df, tokenizer)

    # Create dataloader
    test_dataloader = Experiment.get_dataloader(test_dataset, batch_size=32, shuffle=True)

    # predict the answers for each sample
    print(f"\n>>> Predicting answers for {len(test_dataset)} samples...")

    with torch.no_grad():
        # create dictionary for the output file
        predictions_dict = {}

        # for each batch
        for idxs, tokenized_examples, contexts, _, _, _, _ in tqdm(test_dataloader, total=len(test_dataloader),
                                                                   unit="batch", leave=True, position=0):
            # convert to tensor the tokenized data
            tokenized_examples = dict_to_device(tokenized_examples, device)

            # pop out the offset mapping from the tokenized_examples
            offset_mappings = tokenized_examples.pop("offset_mapping")

            # compute start/end logits using the model
            start_logits, end_logits = model(tokenized_examples)

            # compute probabilities and indexes of the start and end tokens from the output of the neural network
            start_indexes = F.log_softmax(start_logits, dim=-1).detach().cpu()
            start_indexes = start_indexes.argmax(dim=-1)
            end_indexes = F.log_softmax(end_logits, dim=-1).detach().cpu()
            end_indexes = end_indexes.argmax(dim=-1)

            # extract the predicted answer from the original context (predictions)
            predictions = input_ids_to_list_strs(contexts, offset_mappings, start_indexes, end_indexes)

            # add to output dictionary
            # empty predictions are solved inside
            # compute_exact and compute_f1 in evaluate.py
            predictions_dict.update(group_predictions_by_question_id(idxs, predictions))

    # save json file
    with open(sys.path[0] + "/data/predictions.json", 'w') as predictions_json:
        json.dump(predictions_dict, predictions_json, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
