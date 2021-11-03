import json


class SQuADModelConfig:
    """
    A class which describes the configuration of the model

    Attributes
    ----------
    pretrained_model: str
        huggingface pretrained model to use
    """

    def __init__(self, json_file_path: str) -> None:
        """
        Constructor method of the class

        Parameters
        ----------
        json_file_path: str
            the path of the .json file of the configuration
        """

        # open the json file in read only mode
        with open(json_file_path, 'r') as json_file:
            # load the json content from the file as a dictionary
            json_content_dict = json.load(json_file)

        # set the class parameters of the configuration
        self.__dict__.update(json_content_dict)

    def __str__(self) -> str:
        """
        Creates a string representation of the class

        Returns
        -------
        str
            a string representation of the configuration
        """

        str_repr = str(self.__dict__)

        return str_repr
