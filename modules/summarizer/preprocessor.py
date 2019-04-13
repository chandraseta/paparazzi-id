import numpy as np

from modules.constants import Constants
from modules.sentence_embedding import SentenceEmbedding
from modules.utils.indosum_util import get_articles_summaries_indices


class Preprocessor:

    @staticmethod
    def calculate_sentence_embedding(
            sentence: str,
            sentence_embedding: SentenceEmbedding
    ) -> np.ndarray:
        return sentence_embedding.calculate_vector_avg(sentence)

    @staticmethod
    def calculate_sentence_embedding_by_article(
            sentences: [str],
            articles_length_limit: int,
            sentence_embedding: SentenceEmbedding
    ) -> [np.ndarray]:
        sentence_vectors = []
        n_sentence = 0
        for sentence in sentences:
            sentence_vectors.append(sentence_embedding.calculate_vector_avg(sentence))
            n_sentence += 1

        while n_sentence < articles_length_limit:
            sentence_vectors.append(np.zeros((Constants.WORD_EMBEDDING_DIMENSION,)))
            n_sentence += 1

        return sentence_vectors

    @staticmethod
    def load_indosum_data(
            sentence_embedding: SentenceEmbedding,
            type: [str],
            articles_length_limit: int = 80,
            summaries_length_limit: int = 13
    ) -> ([str], [[np.array]]):
        articles, _, gold_labels = get_articles_summaries_indices(data_types=type)

        filtered_articles = []
        filtered_gold_labels = []

        for index, article in enumerate(articles):
            if len(article) > articles_length_limit or len(gold_labels) > summaries_length_limit:
                filtered_articles.append(article)
                filtered_gold_labels.append(gold_labels[index])

        # Calculate sentence vector and add padding
        vectorized_articles = []
        for article in filtered_articles:
            vectorized_article = Preprocessor.calculate_sentence_embedding_by_article(
                article, articles_length_limit, sentence_embedding
            )
            vectorized_articles.append(vectorized_article)

            # DEBUG
            break

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

            while n_labels < articles_length_limit:
                ohe_label.append(np.array([1, 0]))
                n_labels += 1

            encoded_gold_labels.append(ohe_label)

            # DEBUG
            break

        return vectorized_articles, encoded_gold_labels

    @staticmethod
    def load_indosum_data_by_sentence(
            sentence_embedding: SentenceEmbedding,
            type: [str],
            articles_length_limit: int = 80,
            summaries_length_limit: int = 13
    ) -> (np.ndarray, np.ndarray):
        articles, _, gold_labels = get_articles_summaries_indices(data_types=type)

        filtered_articles = []
        filtered_gold_labels = []

        for index, article in enumerate(articles):
            if len(article) > articles_length_limit or len(gold_labels) > summaries_length_limit:
                filtered_articles.append(article)
                filtered_gold_labels.append(gold_labels[index])

        # Calculate sentence vector and add padding
        vectorized_sentences = []
        for article in filtered_articles:
            for sentence in article:
                vectorized_sentence = Preprocessor.calculate_sentence_embedding(
                    sentence, sentence_embedding
                )
                vectorized_sentences.append(vectorized_sentence)

                # DEBUG
                break

        # One-Hot encode gold labels
        encoded_gold_labels = []
        for article_label in filtered_gold_labels:
            for sentence_label in article_label:
                if sentence_label == 0:
                    encoded_gold_labels.append(np.array([1, 0]))
                else:
                    encoded_gold_labels.append(np.array([0, 1]))

                # DEBUG
                break

        np_sentences = np.array(vectorized_sentences)
        np_labels = np.array(encoded_gold_labels)

        return np_sentences, np_labels