import torch
import torch.nn as nn
from transformers import AutoModel
import os
import torch.nn.functional as F

if 'COLAB_TPU_ADDR' in os.environ:
    import torch_xla.core.xla_model as xm


# create a PyTorch module which computes the positional embeddings of the tokens in the given input sequence
class PositionIds(nn.Module):
    """
    PyTorch module that computes the ids of the positional embeddings of a sequence of tokens
    """

    def __init__(self):
        super(PositionIds, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, tokenized_examples):
        """
        Forward method of the model

        Parameters
        ----------
        tokenized_examples
            output of the tokenizer
        """

        # extract the size of the tokenized sentences batch
        input_size = tokenized_examples['input_ids'].size()

        # extract the length of the senteces
        seq_length = input_size[1]

        # return the positional ids of the input sentences
        return torch.arange(seq_length, dtype=torch.long).unsqueeze(0).expand(input_size).to(self.dummy_param.device)


# define a PyTorch module that can be loaded and saved from/to file
class LoadableModule(nn.Module):
    """
    PyTorch module that can be loaded from file
    """

    def save_model(self,
                   where: str,
                   use_tpu: bool = False) -> None:
        """
        Method that saves the module to file

        Parameter
        ---------
        where: str
            path where to save the module
        use_tpu: bool
            flag which defines the use of the tpu (default is False)
        """

        from inspect import signature
        init_params = signature(self.__init__).parameters

        # get dict with only params in init
        dict_init_values = {key: value for key, value in self.__dict__.items() if key in init_params}

        # create the dict to save
        dict_to_save = {
            "params": dict_init_values,
            "state": self.state_dict(),
            "name": self.__class__.__name__
        }

        # save it
        if use_tpu:
            xm.save(dict_to_save, where)
        else:
            torch.save(dict_to_save, where)

    @staticmethod
    def load_model(path: str,
                   device,
                   use_tpu: bool = False) -> nn.Module:
        """
        Method that loads model from file

        Parameters
        ----------
        path: str
            path of the file to load as module
        device
            device on which the model should be loaded
        use_tpu: bool
            flag which defines the use of the tpu (default is False)
        """

        # load module from file
        if use_tpu:
            info = xm.load(path)
        else:
            info = torch.load(path, map_location=device)

        name = info['name']
        params = info['params']

        cls = globals()[name]

        model = cls(**params)
        model.load_state_dict(info['state'])

        return model


# define baseline model
class BaselineSQuADModel(LoadableModule):
    """
    Baseline model to solve a SQuAD question answering problem
    """

    def __init__(self,
                 model_name: str,
                 dropout_rate: float = 0.3,
                 debug: bool = False,
                 freeze_pre_trained: bool = False) -> None:
        """
        Constructor method of the class

        Parameters
        ----------
        model_name: str
            name of the huggingface pretrained model to use
        dropout_rate: float
            dropout rate of the droupout layers (default 0.3)
        debug: bool
            flag which defines the debug mode which allows the print of the shapes (default is False)
        freeze_pre_trained: bool
            flag which defines the freezing of the pretrained layer (default is True)
        """

        # call constructor of the super class
        super(BaselineSQuADModel, self).__init__()

        self.debug = debug
        self.model_name = model_name

        # define pretrained layer
        self.pre_trained = AutoModel.from_pretrained(model_name)

        # freeze pretrained layer if required
        if freeze_pre_trained:
            for param in self.pre_trained.parameters():
                param.requires_grad = False

        self.config = self.pre_trained.config
        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear = nn.Linear(self.config.hidden_size, 2)

        # if the model is not distilbert
        if "distilbert" not in model_name:
            # the model is not distilbert so it requires the positional embeddings
            self.position_ids = PositionIds()
        else:
            # the model is distilbert so it doesn't require the positional embeddings
            self.position_ids = None

    def forward(self, tokenized_examples) -> tuple:
        """
        Forward method of the baseline model

        Parameters
        ----------
        tokenized_examples
            output of the tokenizer

        Returns
        -------
        tuple
            tuple containing
                - the predicted start index logits
                - the predicted end index logits
        """

        # if the model is not distilbert
        if not self.position_ids:
            # remove sentence embeddings from the tokenizer output
            tokenized_examples.pop('token_type_ids')

            # compute the output of the pretrained layer
            pre_trained_outputs = self.pre_trained(
                **tokenized_examples
            )

        else:
            # compute positional embeddings
            position_ids = self.position_ids(tokenized_examples)

            # compute the output of the pretrained layer
            pre_trained_outputs = self.pre_trained(
                **tokenized_examples,
                position_ids=position_ids
            )

        # extract sequence output from the pretrained output
        sequence_output = pre_trained_outputs[0]

        # sequence_output size: (batch_size, max_length_model, hidden_size_model)
        if self.debug:
            print("Sequence output size:", sequence_output.size())

        # dropout
        sequence_output = self.dropout(sequence_output)

        # compute start/end logits
        logits = self.linear(sequence_output)

        # logits size: (batch_size, max_length_model, 2), where 2 is the number of output labels (start and end)

        if self.debug:
            print(f"Logits size: {logits.size()}")

        # split output of the network along last dimension to get start and end label logits
        start_logits, end_logits = logits.split(1, dim=-1)

        # start_logits size: (batch_size, max_length_model, 1)
        # end_logits size: (batch_size, max_length_model, 1)

        # remove extra 1-size dimensions
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # start_logits size: (batch_size, max_length_model)
        # end_logits size: (batch_size, max_length_model)

        if self.debug:
            print(f"Start logits size: {start_logits.size()}")
            print(f"End logits size: {start_logits.size()}")

        return start_logits, end_logits


# define highway network
class Highway(nn.Module):
    """
    Define a highway network
    """

    def __init__(self, size, num_layers, f):
        super(Highway, self).__init__()

        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.f = f

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear

        return x


# define enhanced model version 1 (without residual connection)
class EnhancedSQuADModel1(LoadableModule):
    """
    Enhanced model to solve a SQuAD question answering problem
    """

    def __init__(self,
                 model_name: str,
                 dropout_rate: float = 0.3,
                 debug: bool = False,
                 freeze_pre_trained: bool = False) -> None:
        """
        Constructor method of the class

        Parameters
        ----------
        model_name: str
            name of the huggingface pretrained model to use
        dropout_rate: float
            dropout rate of the droupout layers (default 0.3)
        debug: bool
            flag which defines the debug mode which allows the print of the shapes (default is False)
        freeze_pre_trained: bool
            flag which defines the freezing of the pretrained layer (default is True)
        """

        # call constructor of the super class
        super(EnhancedSQuADModel1, self).__init__()

        self.debug = debug
        self.model_name = model_name

        # define pretrained layer
        self.pre_trained = AutoModel.from_pretrained(model_name)

        # freeze pretrained layer if required
        if freeze_pre_trained:
            for param in self.pre_trained.parameters():
                param.requires_grad = False

        self.config = self.pre_trained.config
        self.encoder = nn.LSTM(self.config.hidden_size, num_layers=2, hidden_size=int(self.config.hidden_size / 2),
                               bidirectional=True, batch_first=True, dropout=dropout_rate)
        self.decoder = nn.LSTM(self.config.hidden_size, num_layers=2, hidden_size=int(self.config.hidden_size / 2),
                               bidirectional=True, batch_first=True, dropout=dropout_rate)
        self.highway = Highway(self.config.hidden_size, num_layers=2, f=F.relu)
        self.linear = nn.Linear(self.config.hidden_size, 2)

        # if the model is not distilbert
        if "distilbert" not in model_name:
            # the model is not distilbert so it requires the positional embeddings
            self.position_ids = PositionIds()
        else:
            # the model is distilbert so it doesn't require the positional embeddings
            self.position_ids = None

    def forward(self, tokenized_examples) -> tuple:
        """
        Forward method of the enhanced model

        Parameters
        ----------
        tokenized_examples
            output of the tokenizer

        Returns
        -------
        tuple
            tuple containing
                - the predicted start index logits
                - the predicted end index logits
        """

        # if the model is not distilbert
        if not self.position_ids:
            # remove sentence embeddings from the tokenizer output
            tokenized_examples.pop('token_type_ids')

            # compute the output of the pretrained layer
            pre_trained_outputs = self.pre_trained(
                **tokenized_examples
            )

        else:
            # compute positional embeddings
            position_ids = self.position_ids(tokenized_examples)

            # compute the output of the pretrained layer
            pre_trained_outputs = self.pre_trained(
                **tokenized_examples,
                position_ids=position_ids
            )

        # extract sequence output from the pretrained output
        sequence_output = pre_trained_outputs[0]

        # sequence_output size: (batch_size, max_length_model, hidden_size_model)

        if self.debug:
            print("Sequence output size:", sequence_output.size())

        # encoder LSTM
        encoder_output, _ = self.encoder(sequence_output)

        # encoder_output size: (batch_size, max_length_model, hidden_size_model)

        if self.debug:
            print("Encoder output size:", encoder_output.size())

        # decoder LSTM
        decoder_output, _ = self.decoder(encoder_output)

        # decoder_output size: (batch_size, max_length_model, hidden_size_model)

        if self.debug:
            print("Decoder output size:", decoder_output.size())

        # highway network
        highway_output = self.highway(decoder_output)

        # highway_output size: (batch_size, max_length_model, hidden_size_model)

        if self.debug:
            print("Highway output size:", highway_output.size())

        # compute start/end logits
        logits = self.linear(highway_output)

        # logits size: (batch_size, max_length_model, 2), where 2 is the number of output labels (start and end)

        if self.debug:
            print(f"Logits size: {logits.size()}")

        # split output of the network along last dimension to get start and end label logits
        start_logits, end_logits = logits.split(1, dim=-1)

        # start_logits size: (batch_size, max_length_model, 1)
        # end_logits size: (batch_size, max_length_model, 1)

        # remove extra 1-size dimensions
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # start_logits size: (batch_size, max_length_model)
        # end_logits size: (batch_size, max_length_model)

        if self.debug:
            print(f"Start logits size: {start_logits.size()}")
            print(f"End logits size: {start_logits.size()}")

        return start_logits, end_logits


# define second enhanced model (with residual connection)
class EnhancedSQuADModel2(LoadableModule):
    """
    Enhanced model to solve a SQuAD question answering problem with residual connection
    """

    def __init__(self,
                 model_name: str,
                 dropout_rate: float = 0.3,
                 debug: bool = False,
                 freeze_pre_trained: bool = False) -> None:
        """
        Constructor method of the class

        Parameters
        ----------
        model_name: str
            name of the huggingface pretrained model to use
        dropout_rate: float
            dropout rate of the droupout layers (default 0.3)
        debug: bool
            flag which defines the debug mode which allows the print of the shapes (default is False)
        freeze_pre_trained: bool
            flag which defines the freezing of the pretrained layer (default is True)
        """

        # call constructor of the super class
        super(EnhancedSQuADModel2, self).__init__()

        self.debug = debug
        self.model_name = model_name

        # define pretrained layer
        self.pre_trained = AutoModel.from_pretrained(model_name)

        # freeze pretrained layer if required
        if freeze_pre_trained:
            for param in self.pre_trained.parameters():
                param.requires_grad = False

        self.config = self.pre_trained.config
        self.encoder = nn.LSTM(self.config.hidden_size, num_layers=2, hidden_size=int(self.config.hidden_size / 2),
                               bidirectional=True, batch_first=True, dropout=dropout_rate)
        self.decoder = nn.LSTM(self.config.hidden_size, num_layers=2, hidden_size=int(self.config.hidden_size / 2),
                               bidirectional=True, batch_first=True, dropout=dropout_rate)
        self.highway = Highway(self.config.hidden_size, num_layers=2, f=F.relu)
        self.linear = nn.Linear(self.config.hidden_size, 2)

        # if the model is not distilbert
        if "distilbert" not in model_name:
            # the model is not distilbert so it requires the positional embeddings
            self.position_ids = PositionIds()
        else:
            # the model is distilbert so it doesn't require the positional embeddings
            self.position_ids = None

    def forward(self, tokenized_examples) -> tuple:
        """
        Forward method of the enhanced model

        Parameters
        ----------
        tokenized_examples
            output of the tokenizer

        Returns
        -------
        tuple
            tuple containing
                - the predicted start index logits
                - the predicted end index logits
        """

        # if the model is not distilbert
        if not self.position_ids:
            # remove sentence embeddings from the tokenizer output
            tokenized_examples.pop('token_type_ids')

            # compute the output of the pretrained layer
            pre_trained_outputs = self.pre_trained(
                **tokenized_examples
            )

        else:
            # compute positional embeddings
            position_ids = self.position_ids(tokenized_examples)

            # compute the output of the pretrained layer
            pre_trained_outputs = self.pre_trained(
                **tokenized_examples,
                position_ids=position_ids
            )

        # extract sequence output from the pretrained output
        sequence_output = pre_trained_outputs[0]

        # sequence_output size: (batch_size, max_length_model, hidden_size_model)

        if self.debug:
            print("Sequence output size:", sequence_output.size())

        # encoder LSTM
        encoder_output, _ = self.encoder(sequence_output)

        # encoder_output size: (batch_size, max_length_model, hidden_size_model)

        if self.debug:
            print("Encoder output size:", encoder_output.size())

        # decoder LSTM
        decoder_output, _ = self.decoder(encoder_output)

        # decoder_output size: (batch_size, max_length_model, hidden_size_model)

        if self.debug:
            print("Decoder output size:", decoder_output.size())

        # residual connection
        residual_output = sequence_output + decoder_output

        # highway network
        highway_output = self.highway(residual_output)

        # highway_output size: (batch_size, max_length_model, hidden_size_model)

        if self.debug:
            print("Highway output size:", highway_output.size())

        # compute start/end logits
        logits = self.linear(highway_output)

        # logits size: (batch_size, max_length_model, 2), where 2 is the number of output labels (start and end)

        if self.debug:
            print(f"Logits size: {logits.size()}")

        # split output of the network along last dimension to get start and end label logits
        start_logits, end_logits = logits.split(1, dim=-1)

        # start_logits size: (batch_size, max_length_model, 1)
        # end_logits size: (batch_size, max_length_model, 1)

        # remove extra 1-size dimensions
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # start_logits size: (batch_size, max_length_model)
        # end_logits size: (batch_size, max_length_model)

        if self.debug:
            print(f"Start logits size: {start_logits.size()}")
            print(f"End logits size: {start_logits.size()}")

        return start_logits, end_logits
