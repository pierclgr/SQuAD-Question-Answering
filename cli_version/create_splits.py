import argparse
import os
from pathlib import Path

from src.config import SQuADModelConfig
from src.dataset import SQuADJSONDataset


# define main method
def main() -> None:
    """
    Main method of the script that splits the dataset given in input into train, val and test splits
    """

    # parse the argument containing the path of the json file to load
    parser = argparse.ArgumentParser()
    parser.add_argument("set", help="Path of the set json file to split")
    parser.add_argument("config", help="Path of the configuration json file")
    args = parser.parse_args()
    json_file_path = args.set
    config_path = args.config

    # create model configuration
    config = SQuADModelConfig(config_path)

    # extract the path of the directory of the input file
    data_folder = os.path.dirname(json_file_path)
    data_folder = Path(data_folder)

    # create the SQuADJSONDataset object related to the json file
    print("\n>>> Loading the .json file...")

    json_datset = SQuADJSONDataset.from_path(json_file_path)

    # create the three splits SQuADJSONDataset objects
    print("\n>>> Splitting...")

    train_json, val_json, test_json = json_datset.split(val_split=config.val_split, test_split=config.test_split)

    # save the three splits to a .json file in the same directory of the input file
    print("\n>>> Saving to file...")

    train_json.dump(data_folder / "train.json")
    val_json.dump(data_folder / "val.json")
    test_json.dump(data_folder / "test.json")

    # remove the answers from the test set
    print("\n>>> Removing answers from test set...")

    test_json.clean_answers()

    # dump the test set cleaned from answers
    print("\n>>> Saving to file...")

    test_json.dump(data_folder / "test_no_answers.json")

    print()


# execute main method
if __name__ == "__main__":
    main()
