import glob
import json
import re


INDOSUM_FILEPATH = 'data/indosum/'
INDOSUM_EXTENSION = '.jsonl'

NO_LEADING_SPACE_SYMBOLS = [',', '.', '!', '?']


def join_tokens(tokens: [str]) -> str:
    sentence = ''

    is_closing_bracket = False
    is_closing_quote = False
    skip_once = False

    for token in tokens:
        token = token.strip()

        if is_closing_bracket and token == ')':
            sentence = sentence[:-1]
            is_closing_bracket = False

        if is_closing_quote and token == '"':
            sentence = sentence[:-1]
            is_closing_quote = False
            skip_once = True

        if token in NO_LEADING_SPACE_SYMBOLS:
            sentence = sentence[:-1]

        sentence += token + ' '

        if token == '(':
            sentence = sentence[:-1]
            is_closing_bracket = True

        if token == '"':
            if not skip_once:
                sentence = sentence[:-1]
                is_closing_quote = True
            else:
                skip_once = False

    sentence = sentence.replace('-', ' - ')
    sentence = re.sub('\s+', ' ', sentence).strip()

    return sentence


def get_articles(indosum_filepath: str = INDOSUM_FILEPATH) -> [[list]]:
    filepaths = glob.glob(indosum_filepath + '*' + INDOSUM_EXTENSION)
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


def get_indices(indosum_filepath: str = INDOSUM_FILEPATH):
    filepaths = glob.glob(indosum_filepath + '*' + INDOSUM_EXTENSION)
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


def get_summaries(indosum_filepath: str = INDOSUM_FILEPATH):
    filepaths = glob.glob(indosum_filepath + '*' + INDOSUM_EXTENSION)
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


def get_articles_summaries_indices(indosum_filepath: str = INDOSUM_FILEPATH):
    filepaths = glob.glob(indosum_filepath + '*' + INDOSUM_EXTENSION)
    articles = []
    summaries = []
    summaries_indices = []

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

            summary = data['summary']
            summary_tokens = []

            for sentence in summary:
                for sentence_token in sentence:
                    summary_tokens.append(sentence_token)

            summaries.append(join_tokens(summary_tokens))

            gold_labels = data['gold_labels']
            indices = []

            current_index = 0
            for paragraph_labels in gold_labels:
                for sentence_labels in paragraph_labels:
                    if sentence_labels:
                        indices.append(current_index)
                    current_index += 1

            summaries_indices.append(indices)

    return articles, summaries, summaries_indices
