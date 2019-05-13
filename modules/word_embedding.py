import gensim
import numpy
import numpy as np
import pandas as pd

from modules.constants import Constants
from modules.utils.text_util import load_stopwords


class WordEmbedding:
    def __init__(self,
            word_embedding_filepath: str = Constants.WORD_EMBEDDING_FILEPATH,
            dimension: int = 300,
    ):
        print('[W2V] Loading word embeddings')
        self._dimension = dimension
        self._word_embeddings_dict = self._load_word_embeddings_dict(word_embedding_filepath)
        print('[W2V] Finished loading sentence embedding')

    def _load_word_embeddings_dict(self, word_embedding_filepath):
        word_embedding_file = open(word_embedding_filepath, 'r')
        word_embeddings = {}
        start_line = True

        for line in word_embedding_file:
            if start_line:
                start_line = False
                continue
            else:
                values = line.split(' ')
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                word_embeddings[word] = coefs

        return word_embeddings

    def word2vec(self, word: str) -> np.ndarray:
        word = word.lower()
        if word in self._word_embeddings_dict:
            return self._word_embeddings_dict[word]
        else:
            return np.zeros((self._dimension,), dtype='float32')

    @property
    def dimension(self):
        return self._dimension

    def calculate_vector_avg(
            self,
            sentence: str,
            remove_stopwords: bool = True
    ) -> np.ndarray:
        sentence_tokens = gensim.utils.simple_preprocess(sentence)
        sentence_vector = np.zeros((self.dimension,))
        n_token = 0

        stopwords = []
        if remove_stopwords:
            stopwords = load_stopwords()

        for token in sentence_tokens:
            if len(token) != 0 and token not in stopwords:
                sentence_vector += self.word2vec(token)
                n_token += 1

        if n_token > 0:
            sentence_vector = sentence_vector / float(n_token)

        return sentence_vector

    def calculate_paragraph_vector_avg(
            self,
            paragraph: [str],
            remove_stopwords: bool = True
    ) -> np.ndarray:
        paragraph_vector = np.zeros((self.dimension,))
        n_token = 0

        stopwords = []
        if remove_stopwords:
            stopwords = load_stopwords()

        for sentence in paragraph:
            sentence_tokens = gensim.utils.simple_preprocess(sentence)

            for token in sentence_tokens:
                if len(token) != 0 and token not in stopwords:
                    paragraph_vector += self.word2vec(token)
                    n_token += 1

        if n_token > 0:
            paragraph_vector = paragraph_vector / float(n_token)

        return paragraph_vector

    def calculate_vector(
            self,
            sentence: str,
            remove_stopwords: bool = True
    ) -> np.ndarray:
        sentence_tokens = gensim.utils.simple_preprocess(sentence)
        sentence_vector = np.zeros((self.dimension,))
        n_token = 0

        stopwords = []
        if remove_stopwords:
            stopwords = load_stopwords()

        for token in sentence_tokens:
            if len(token) != 0 and token not in stopwords:
                sentence_vector += self.word2vec(token)
                n_token += 1

        return sentence_vector

    def calculate_vector_flatten(
            self,
            sentence: str,
            max_length: int,
            remove_stopwords: bool = True
    ) -> np.ndarray:
        sentence_tokens = gensim.utils.simple_preprocess(sentence)
        sentence_vector = np.zeros((0,))
        n_token = 0

        stopwords = []
        if remove_stopwords or len(sentence) > max_length:
            stopwords = load_stopwords()

        for token in sentence_tokens:
            if len(token) != 0 and n_token < max_length and token not in stopwords:
                sentence_vector = np.append(sentence_vector, self.word2vec(token))
                n_token += 1

        # Padding
        while n_token < max_length:
            sentence_vector = np.append(sentence_vector, np.zeros((self.dimension,)))
            n_token += 1
        return sentence_vector

    def calculate_vector_sif(
            self,
            sentence: str,
            remove_stopwords: bool = True
    ) -> np.ndarray:
        # Based on "A Simple but Tough-to-Beat Baseline for Sentence Embeddings
        # https://openreview.net/pdf?id=SyK00v5xx
        pass
