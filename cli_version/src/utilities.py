import re


# function converting a string sentence to a list
def sentence_to_wordlist(text: str) -> list:
    """
    Converts a string sentence to a word list

    Parameters
    ----------
    text: str
        string sentence to convert

    Returns
    -------
    list
        string sentence converted to list of tokens
    """

    # split the sentence into a token list
    word_list = re.split("\s", text)

    return word_list


# function to extract the answer text from the context
def extract_original_answer(context: str,
                            offset_mapping: list,
                            start_index: int,
                            end_index: int) -> str:
    """
    Extracts the answer text from the context using some start and end character indexes

    Parameters
    ----------
    context: str
        the original context text
    offset_mapping: list
        the mapping produced by the tokenizer between the tokenized text and the original text
    start_index: int
        the index of the start token of the answer in the tokenized text
    end_index: int
        the index of the end token of the answer in the tokenized text

    Returns
    -------
    str
        extracted text in the original context
    """
    # compute the start and end character indexes using the mapping produced by the tokenizer
    start_char = offset_mapping[start_index][0]
    end_char = offset_mapping[end_index][1]

    # extract the text using the new computed indexes
    text = context[start_char:end_char]

    return text


def dict_to_device(d, device) -> dict:
    """
    Creates a dictionary on PyTorch device using an input dictionary

    Parameters
    ----------
    d
        input dictionary
    device
        PyTorch device

    Returns
    -------
    dict
        dictionary on the PyTorch device
    """
    return {key: value.to(device) for key, value in d.items()}


def input_ids_to_list_strs(contexts, offset_mappings, start_positions, end_positions) -> list:
    """
    Transform Tensor (Batch of context+question) into list of str (extract the answer from the context+question)

    Parameters
    ----------
    contexts
        contexts
    offset_mappings
        mappings produced by the tokenizer
    start_positions
        starting positions in the contexts
    end_positions
        ending positions in the contexts

    Returns
    -------
    list
        output list from the input tensor
    """

    return [extract_original_answer(contexts[i], offset_mappings[i], start_positions[i], end_positions[i]) for i in
            range(len(contexts))]
