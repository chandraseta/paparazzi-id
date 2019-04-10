import gensim
import numpy as np

from modules.constants import Constants
from modules.word_embedding import WordEmbedding
from modules.utils.text_util import load_stopwords


class SentenceEmbedding:

    def __init__(self):
        self._word_embedding = WordEmbedding(Constants.WORD_EMBEDDING_FILEPATH)

    def calculate_vector_avg(
            self,
            sentence: str,
            remove_stopwords: bool = True
    ) -> np.ndarray:
        sentence_tokens = gensim.utils.simple_preprocess(sentence)
        sentence_vector = np.zeros((self._word_embedding.dimension,))
        n_token = 0

        stopwords = []
        if remove_stopwords:
            stopwords = load_stopwords()

        for token in sentence_tokens:
            if len(token) != 0 and token not in stopwords:
                sentence_vector += self._word_embedding.word2vec(token)
                n_token += 1

        sentence_vector = sentence_vector / float(n_token)
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
                sentence_vector = np.append(sentence_vector, self._word_embedding.word2vec(token))
                n_token += 1

        # Padding
        while n_token < max_length:
            sentence_vector = np.append(sentence_vector, np.zeros((self._word_embedding.dimension,)))
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
