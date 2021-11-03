import json
import random
import math
import pathlib
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.utils.data as data
from transformers import PreTrainedTokenizer


# this class is used to manage the data contained in a .json SQuAD dataset file in order to create a usable dataframe
# from it or to create train/val/test splits if we need them
class SQuADJSONDataset:
    """
    A class to manage the json file which contains the SQuAD dataset

    Attributes
    ----------
    version: str
        version of the dataset
    data: dict
        dictionary containing the data of the dataset
    """

    def __init__(self, json_dict: dict) -> None:
        """
        The constructor of the class

        Parameters
        ----------
        json_dict: dict
            the dict containing the data of the dataset
        """

        # initialize the "version" attribute of the class object containing
        # the version of the dataset
        self.version = json_dict['version']

        # initialize the "data" attribute of the class object containing the
        # data
        self.data = json_dict['data']

    @classmethod
    def from_path(cls, json_file_path: str):
        """
        Static method which loads the data dictionary from the json file

        Parameters
        ----------
        json_file_path: str
            the path of the json file

        Returns
        -------
        SQuADJSONDataset
            SQuADJSONDataset object of the inputted json file
        """

        # open the json file in read only mode
        with open(json_file_path, 'r') as json_file:
            # load the json content from the file as a dictionary
            json_content_dict = json.load(json_file)

        # return the SQuADJSONDataset object of the current json
        return cls(json_content_dict)

    def split(self, val_split: float = 0.15, test_split: float = 0.15, shuffle: bool = True) -> tuple:
        """
        A method which splits the current dataset into train-validation-test
        using the given split ratios; the split will be performed on the titles
        of the dataset arguments

        Parameters
        ----------
        val_split: float
            percentage of the validation split in decimal notation (default is
            0.15)
        test_split: float
            percentage of the test split in decimal notation (default is 0.15)
        shuffle: bool
            if True, shuffle the dataset to extract randomly samples from the
            splits

        Returns
        -------
        SQuADJSONDataset
            training split
        SQuADJSONDataset
            validation split
        SQuADJSONDataset
            testing split
        """

        train_set = self.data

        # shuffle the data randomly
        if shuffle:
            random.shuffle(train_set)

        # compute the index of the test split
        total_samples = len(train_set)
        test_samples = total_samples - math.floor(total_samples * (1 - test_split))
        val_samples = total_samples - math.floor(total_samples * (1 - val_split))

        # split the dataset into train and test
        test_set = train_set[:test_samples]
        train_set = train_set[test_samples:]

        # split the dataset into train and validation
        val_set = train_set[:val_samples]
        train_set = train_set[val_samples:]

        # save the dataset
        train_set = {'data': train_set, 'version': self.version}
        val_set = {'data': val_set, 'version': self.version}
        test_set = {'data': test_set, 'version': self.version}

        # return the three splits
        return SQuADJSONDataset(train_set), SQuADJSONDataset(val_set), SQuADJSONDataset(test_set)

    def dump(self, output_path: pathlib.Path) -> None:
        """
        Save the JSONDataset object into file specified by the input path
        following this structure:
            {
                data: ...,
                version: ...
            }

        Parameters
        ----------
        output_path: Path
            path where to save the data
        """

        # open the output file in write mode to save the content of the object
        with open(output_path, 'w') as fp:
            json_data = {'data': self.data, 'version': self.version}

            # save the in json format in a json file
            json.dump(json_data, fp)

    def clean_answers(self) -> None:
        """
        Removes all the answers from a JSON dataset
        """

        # for each argument in the dataset "data"
        for argument in tqdm(self.data, unit='argument', leave=True, position=0):

            # for each paragraph in the current argument
            for paragraph in argument['paragraphs']:

                # for each question in the current paragraph
                for qas in paragraph['qas']:
                    # remove the answers list
                    del qas['answers']

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert JSONDataset to a pandas DataFrame

        Returns
        -------
        DataFrame
            pandas DataFrame of the JSONDataset
        """

        # initialize an empty dataframe list
        temp = []

        # for each argument in the dataset "data"
        for argument in tqdm(self.data, unit='argument', leave=True, position=0):

            # extract the title of the current argument
            title = argument['title']

            # for each paragraph in the current argument
            for paragraph in argument['paragraphs']:

                # extract the context of the current paragraph
                context = paragraph['context']

                # for each question in the current paragraph
                for qas in paragraph['qas']:

                    # extract the id of the current question
                    id = qas['id']

                    # extract the text of the question
                    question = qas['question']

                    # if the current question has a list of answers
                    if 'answers' in qas:

                        # for each answer
                        for answer in qas['answers']:
                            # extract the index of the starting character of the
                            # current answer in the context
                            answer_start = answer['answer_start']

                            # extract the text of the current answer
                            text = answer['text']

                            # compute the index of the ending character of the
                            # current answer in the context
                            answer_end = answer_start + len(text)

                            # create a dictionary containing all the instances
                            # extracted and append it to the temp list
                            temp.append({
                                "title": title,
                                "context": context,
                                "question": question,
                                "id": id,
                                "answer_start": answer_start,
                                "answer_end": answer_end,
                                "answer_text": text
                            })
                    else:

                        # create a dictionary containing all the instances but
                        # the answers and append it to the temp list
                        temp.append({
                            "title": title,
                            "context": context,
                            "question": question,
                            "id": id
                        })

        # create a pandas DataFrame from the temp list and return it
        df = pd.DataFrame(temp)
        return df


# create the PyTorch dataset class
class SQuADDataset(data.Dataset):
    """
    A PyTorch dataset class for the SQuAD dataset
    """

    def __init__(self,
                 dataframe: pd.DataFrame,
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = None,
                 doc_stride: int = 128) -> None:
        """
        The constructor of the class

        Parameters
        ----------
        dataframe: DataFrame
            pandas DataFrame to transform into a PyTorch dataset
        tokenizer: PreTrainedTokenizer
            the huggingface tokenizer that will be used to tokenize the text
        max_length: int
            max length to use for context and question (default is None)
        doc_stride: int
            the authorized overlap between two part of the context when splitting it is needed (default is 128)
        """

        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.doc_stride = doc_stride

        # if max_length parameter is not specified,
        if not max_length:
            # compute the max length of the dataset using the tokenizer max_length and the doc_stride parameters
            self.max_length = self.tokenizer.model_max_length - self.doc_stride
        else:
            self.max_length = max_length

    def __len__(self) -> int:
        """
        Returns the number of elements in the dataset

        Returns
        -------
        int
            number of elements of the dataset
        """

        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple:
        """
        Returns an item from the dataset at the corresponding input index

        Parameters
        ----------
        index: int
            index of the row in the dataframe to extact

        Returns
        -------
        tuple
            tuple containing
                - question id
                - context
                - question
                - index of the answer start character
                - index of the answer end character
                - text of the answer
        """

        # extract selected row from dataframe
        row = self.dataframe.iloc[index]

        # extract elements from the row
        idx = row.id
        context = row.context
        question = row.question

        if 'answer_text' in self.dataframe.columns:
            answer_start = row.answer_start
            answer_end = row.answer_end
            answer_text = row.answer_text
        else:
            answer_start = None
            answer_end = None
            answer_text = None

        # return tuple containing the elements extracted from the row
        return idx, context, question, answer_start, answer_end, answer_text

    @staticmethod
    def compute_start_end_index(tokenized_examples,
                                idxs: list,
                                contexts: list,
                                questions: list,
                                answer_starts: list,
                                answer_ends: list,
                                answer_texts: list,
                                tokenizer: PreTrainedTokenizer) -> tuple:
        """
        Method which computes the indexes of the start and end tokens of the answer in the tokenized text using the
        original character indexes

        Parameters
        ----------
        tokenized_examples
            data tokenized by the tokenizer
        idxs: list
            list of original id of each question
        contexts: list
            list of original contexts
        questions: list
            list of original questions
        answer_starts: list
            list of original start characters indexes
        answer_ends: list
            list of original end characters indexes
        answer_texts: list
            list of original texts of the answer
        tokenizer: PreTrainedTokenizer
            huggingface pretrained tokenizer
        Returns
        -------
        tuple
            tuple containing:
                - the ids
                - the text of the contexts
                - the text of the questions
                - the new computed start token indexes
                - the new computed end token indexes
                - the text of the answers
        """

        if all(texts is None for texts in answer_texts):
            contains_answers = False
        else:
            contains_answers = True

        # extract and remove from the tokenized data the mappings that the tokenizer produced for us that we are going
        # to use to compute the token index
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples["offset_mapping"]

        # initialize the new indexes lists to empty
        starts = []
        ends = []

        # initialize the new indexes, contexts, questions and answer_texts lists to empty
        new_idxs = []
        new_contexts = []
        new_questions = []
        new_answer_texts = []

        # for each mapping in the offset_mapping
        for i, offsets in enumerate(offset_mapping):
            # extract the ids of the tokens of the current sample from the tokenized examples
            input_ids = tokenized_examples["input_ids"][i]

            # extract the index of the cls_token in the vocabulary of the tokenizer
            cls_index = input_ids.tolist().index(tokenizer.cls_token_id)

            # grab the sequence corresponding to that example (to know what is the context and what is the question)
            sequence_ids = tokenized_examples.sequence_ids(i)

            # one example can give several spans, this is the index of the example containing this span of text
            sample_index = sample_mapping[i]

            # grab the id, answer text, start index, end index, context and question of the current sample
            idx = idxs[sample_index]
            context = contexts[sample_index]
            question = questions[sample_index]

            if contains_answers:

                answer_text = answer_texts[sample_index]
                answer_start = answer_starts[sample_index]
                answer_end = answer_ends[sample_index]

                # if no answers are given, set the cls_index as answer
                if len(answer_text) == 0:
                    starts.append(cls_index)
                    ends.append(cls_index)
                else:
                    # start/end character index of the answer in the text
                    start_char = answer_start
                    end_char = answer_end

                    # start token index of the current span in the text
                    token_start_index = 0
                    while sequence_ids[token_start_index] != 1:
                        token_start_index += 1

                    # end token index of the current span in the text
                    token_end_index = len(input_ids) - 1
                    while sequence_ids[token_end_index] != 1:
                        token_end_index -= 1

                    # detect if the answer is out of the span (in which case this feature is labeled with the CLS index)
                    if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                        starts.append(cls_index)
                        ends.append(cls_index)
                    else:
                        # otherwise move the token_start_index and token_end_index to the two ends of the answer
                        # we could go after the last offset if the answer is the last word (edge case)
                        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                            token_start_index += 1
                        starts.append(token_start_index - 1)
                        while offsets[token_end_index][1] >= end_char:
                            token_end_index -= 1
                        ends.append(token_end_index + 1)

                # add answer to the list
                new_answer_texts.append(answer_text)

            # add the id, context and question
            new_idxs.append(idx)
            new_contexts.append(context)
            new_questions.append(question)

        return new_idxs, new_contexts, new_questions, starts, ends, new_answer_texts

    def collate_fn(self, batch: list) -> tuple:
        """
        Collate function for the DataLoader

        Parameters
        ----------
        batch: list of tuples
            batch of elements randomly extracted

        Returns
        -------
        tuple
            tuple containing:
                - batch of ids: list
                - batch of tokenized data
                - batch of contexts: list
                - batch of questions: list
                - batch of start token indexes: torch.Tensor
                - batch of end token indexes: torch.Tensor
                - batch of answer texts: list
        """

        # unzip the batch
        idxs, contexts, questions, answer_starts, answer_ends, answer_texts = zip(*batch)

        # convert the batch of questions and contexts to a list
        questions = list(questions)
        contexts = list(contexts)

        # tokenize the contexts and questions and concatenate together into a single sample
        tokenized_examples = self.tokenizer(
            questions,
            contexts,
            add_special_tokens=True,  # add [CLS] and [SEP] tokens
            max_length=self.max_length,  # max possible length of a feature
            stride=self.doc_stride,  # number of overlapping tokens
            return_overflowing_tokens=True,  # return also the features truncated and overlapped
            padding="max_length",  # pad each feature to max_length
            truncation="only_second",  # truncate only the second feature (the context)
            return_offsets_mapping=True,
            # return for each original input token the corresponding start and end character index
            return_attention_mask=True,  # return the mask hiding the padding
            return_token_type_ids=True,  # return also segment embeddings
            return_tensors='pt'  # return PyTorch tensors
        )

        # compute the indexes of the start and end tokens of the answer in the tokenized text
        idxs, contexts, questions, starts, ends, answer_texts = SQuADDataset.compute_start_end_index(
            tokenized_examples,
            idxs,
            contexts,
            questions,
            answer_starts,
            answer_ends,
            answer_texts,
            self.tokenizer)

        # convert to tensor the lists of start and end indexes
        starts = torch.as_tensor(starts, dtype=torch.long)
        ends = torch.as_tensor(ends, dtype=torch.long)

        return idxs, tokenized_examples, contexts, questions, starts, ends, answer_texts
