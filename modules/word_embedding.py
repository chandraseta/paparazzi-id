import numpy as np
import pandas as pd


class WordEmbedding:
    def __init__(self,
            word_embedding_filepath: str,
            dimension: int = 300,
            # train: bool = False,
            # training_sentences_filepaths: [str] = []
    ):
        self._dimension = dimension
        self._word_embeddings = self._load_word_embeddings(word_embedding_filepath)
        self._word_embeddings_dict = self._load_word_embeddings_dict(word_embedding_filepath)
        # TODO: add train functionality on WordEmbedding class or a separate class
            

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
        if word in self._word_embeddings:
            return self._word_embeddings[word]
        else:
            return np.zeros((self._dimension,), dtype='float32')