import glob
import json
import re

from modules.constants import Constants


def join_tokens(tokens: [str]) -> str:
    sentence = ''

    is_closing_bracket = False
    is_closing_double_quote = False
    is_closing_single_quote = False
    skip_once_double_quote = False
    skip_once_single_quote = False

    for token in tokens:
        token = token.strip()
        token = token.replace('“', '"')
        token = token.replace('”', '"')
        token = token.replace('---', '-')
        token = token.replace('--', '-')

        if is_closing_bracket and token == ')':
            sentence = sentence[:-1]
            is_closing_bracket = False

        if is_closing_double_quote and token == '"':
            sentence = sentence[:-1]
            is_closing_double_quote = False
            skip_once_double_quote = True

        if is_closing_single_quote and token == '\'':
            sentence = sentence[:-1]
            is_closing_single_quote = False
            skip_once_single_quote = True

        if token in Constants.NO_LEADING_SPACE_SYMBOLS:
            sentence = sentence[:-1]

        sentence += token + ' '

        if token == '(':
            sentence = sentence[:-1]
            is_closing_bracket = True

        if token == '"':
            if not skip_once_double_quote:
                sentence = sentence[:-1]
                is_closing_double_quote = True
            else:
                skip_once_double_quote = False

        if token == '\'':
            if not skip_once_single_quote:
                sentence = sentence[:-1]
                is_closing_single_quote = True
            else:
                skip_once_single_quote = False

    sentence = sentence.replace('-', ' - ')
    sentence = re.sub('\s+', ' ', sentence).strip()

    return sentence


def get_articles(indosum_filepath: str = Constants.INDOSUM_FILEPATH) -> [[list]]:
    filepaths = glob.glob(indosum_filepath + '*' + Constants.INDOSUM_EXTENSION)
    articles = []

    for filepath in filepaths:
        file = open(filepath, 'r')

        for line in file:
            data = json.loads(line)

            paragraphs = data['paragraphs']
            sentence_tokens = []

            for paragraph in paragraphs:
                for sentence in paragraph:
                    for sentence_token in sentence:
                        sentence_tokens.append(sentence_token)

            articles.append(join_tokens(sentence_tokens))

    return articles


def get_summaries(indosum_filepath: str = Constants.INDOSUM_FILEPATH):
    filepaths = glob.glob(indosum_filepath + '*' + Constants.INDOSUM_EXTENSION)
    summaries = []

    for filepath in filepaths:
        file = open(filepath, 'r')

        for line in file:
            data = json.loads(line)

            summary = data['summary']
            summary_tokens = []

            for sentence in summary:
                for sentence_token in sentence:
                    summary_tokens.append(sentence_token)

            summaries.append(join_tokens(summary_tokens))

    return summaries


def get_indices(indosum_filepath: str = Constants.INDOSUM_FILEPATH):
    filepaths = glob.glob(indosum_filepath + '*' + Constants.INDOSUM_EXTENSION)
    summaries_indices = []

    for filepath in filepaths:
        file = open(filepath, 'r')

        for line in file:
            data = json.loads(line)

            gold_labels = data['gold_labels']
            indices = []

            current_index = 0
            for paragraph_labels in gold_labels:
                for sentence_labels in paragraph_labels:
                    if sentence_labels:
                        indices.append(current_index)

                    current_index += 1

            summaries_indices.append(indices)

    return summaries_indices


def get_gold_labels_length(indosum_filepath: str = Constants.INDOSUM_FILEPATH, data_types: [str] = None):
    if data_types is None:
        data_types = ['train', 'dev', 'test']

    filepaths = []
    for data_type in data_types:
        filepaths.extend(glob.glob(indosum_filepath + data_type + '*' + Constants.INDOSUM_EXTENSION))

    gold_labels_length = []
    for filepath in filepaths:
        file = open(filepath, 'r')

        for line in file:
            data = json.loads(line)

            gold_labels = data['gold_labels']

            counter = 0
            for paragraph_labels in gold_labels:
                for sentence_labels in paragraph_labels:
                    if sentence_labels:
                        counter += 1

            gold_labels_length.append(counter)

    return gold_labels_length


def get_articles_by_sentences(indosum_filepath: str = Constants.INDOSUM_FILEPATH):
    filepaths = glob.glob(indosum_filepath + '*' + Constants.INDOSUM_EXTENSION)
    articles = []

    for filepath in filepaths:
        file = open(filepath, 'r')

        for line in file:
            data = json.loads(line)

            data_paragraphs = data['paragraphs']
            paragraphs = []

            for paragraph in data_paragraphs:
                for sentence in paragraph:
                    sentence_tokens = []
                    for sentence_token in sentence:
                        sentence_tokens.append(sentence_token)
                    joined_sent = join_tokens(sentence_tokens)
                    paragraphs.append(joined_sent)

            articles.append(paragraphs)

    return articles


def get_articles_summaries_indices(
        indosum_filepath: str = Constants.INDOSUM_FILEPATH,
        data_types: [str] = None
) -> ([[str]], [[str]], [[int]]):
    if data_types is None:
        data_types = ['train', 'dev', 'test']

    filepaths = []
    for data_type in data_types:
        filepaths.extend(glob.glob(indosum_filepath + data_type + '*' + Constants.INDOSUM_EXTENSION))

    articles = []
    summaries = []
    summaries_indices = []

    for filepath in filepaths:
        file = open(filepath, 'r')

        for line in file:
            data = json.loads(line)

            data_paragraphs = data['paragraphs']
            paragraphs = []

            for paragraph in data_paragraphs:
                for sentence in paragraph:
                    sentence_tokens = []
                    for sentence_token in sentence:
                        sentence_tokens.append(sentence_token)
                    joined_sent = join_tokens(sentence_tokens)
                    paragraphs.append(joined_sent)

            articles.append(paragraphs)

            summary = data['summary']
            summary_tokens = []

            for sentence in summary:
                for sentence_token in sentence:
                    summary_tokens.append(sentence_token)
            joined_summary = join_tokens(summary_tokens)
            summaries.append(joined_summary)

            gold_labels = data['gold_labels']
            indices = []

            for paragraph_labels in gold_labels:
                for sentence_labels in paragraph_labels:
                    if sentence_labels:
                        indices.append(1)
                    else:
                        indices.append(0)

            summaries_indices.append(indices)

    return articles, summaries, summaries_indices

