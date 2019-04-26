import numpy as np

from modules.constants import Constants
from modules.word_embedding import WordEmbedding
from modules.utils.indosum_util import get_articles_summaries_indices


class Preprocessor:

    @staticmethod
    def calculate_sentence_vector(
            sentence: str,
            word_embedding: WordEmbedding
    ) -> np.ndarray:
        return word_embedding.calculate_vector_avg(sentence)

    @staticmethod
    def calculate_sentence_vectors_by_article(
            sentences: [str],
            word_embedding: WordEmbedding
    ) -> [np.ndarray]:
        sentence_vectors = []
        n_sentence = 0
        for sentence in sentences:
            sentence_vectors.append(word_embedding.calculate_vector_avg(sentence))
            n_sentence += 1

        while n_sentence < Constants.MAXIMUM_SENTENCE_LENGTH:
            sentence_vectors.append(np.zeros((Constants.WORD_EMBEDDING_DIMENSION,)))
            n_sentence += 1

        return sentence_vectors

    @staticmethod
    def load_indosum_data(word_embedding: WordEmbedding, data_type: [str]) -> ([str], [[np.array]]):
        articles, _, gold_labels = get_articles_summaries_indices(data_types=data_type)

        filtered_articles = []
        filtered_gold_labels = []

        for index, article in enumerate(articles):
            if len(article) <= Constants.MAXIMUM_SENTENCE_LENGTH:
                filtered_articles.append(article)
                filtered_gold_labels.append(gold_labels[index])

        # Calculate sentence vector and add padding
        vectorized_articles = []
        for article in filtered_articles:
            vectorized_article = Preprocessor.calculate_sentence_vectors_by_article(
                article, word_embedding
            )
            vectorized_articles.append(vectorized_article)

        # One-Hot encode gold labels
        encoded_gold_labels = []
        for article_label in filtered_gold_labels:
            ohe_label = []
            n_labels = 0
            for sentence_label in article_label:
                if sentence_label == 0:
                    ohe_label.append(np.array([1, 0]))
                else:
                    ohe_label.append(np.array([0, 1]))
                n_labels += 1

            while n_labels < Constants.MAXIMUM_SENTENCE_LENGTH:
                ohe_label.append(np.array([1, 0]))
                n_labels += 1

            encoded_gold_labels.append(ohe_label)

        vectorized_articles = np.array(vectorized_articles)
        encoded_gold_labels = np.array(encoded_gold_labels)

        return vectorized_articles, encoded_gold_labels

    @staticmethod
    def load_indosum_data_by_sentence(
            word_embedding: WordEmbedding,
            type: [str]
    ) -> (np.ndarray, np.ndarray):
        articles, _, gold_labels = get_articles_summaries_indices(data_types=type)

        filtered_articles = []
        filtered_gold_labels = []

        for index, article in enumerate(articles):
            if len(article) <= Constants.MAXIMUM_SENTENCE_LENGTH:
                filtered_articles.append(article)
                filtered_gold_labels.append(gold_labels[index])

        # Calculate sentence vector and add padding
        vectorized_sentences = []
        for article in filtered_articles:
            for sentence in article:
                vectorized_sentence = Preprocessor.calculate_sentence_vector(
                    sentence, word_embedding
                )
                vectorized_sentences.append(vectorized_sentence)

        # One-Hot encode gold labels
        encoded_gold_labels = []
        for article_label in filtered_gold_labels:
            for sentence_label in article_label:
                if sentence_label == 0:
                    encoded_gold_labels.append(np.array([1, 0]))
                else:
                    encoded_gold_labels.append(np.array([0, 1]))

        vectorized_sentences = np.array(vectorized_sentences)
        encoded_gold_labels = np.array(encoded_gold_labels)

        return vectorized_sentences, encoded_gold_labels

    @staticmethod
    def preprocess_text(sentences: [str], word_embedding: WordEmbedding) -> np.ndarray:
        if len(sentences) > Constants.MAXIMUM_SENTENCE_LENGTH:
            sentences = sentences[:Constants.MAXIMUM_SENTENCE_LENGTH]

        sentences_vectors = []
        sentences_vector = Preprocessor.calculate_sentence_vectors_by_article(
            sentences, word_embedding
        )

        sentences_vectors.append(sentences_vector)
        sentences_vectors = np.array(sentences_vectors)

        return sentences_vectors
