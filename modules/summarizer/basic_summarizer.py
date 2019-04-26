import networkx as nx
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from modules.constants import Constants
from modules.utils.text_util import split_to_sentences
from modules.word_embedding import WordEmbedding


class BasicSummarizer:
    # TODO: Find out about multi-document summarization

    def __init__(self, word_embedding: WordEmbedding):
        self._word_embedding = word_embedding

    def textrank_avg(self, text, n_sentence: int = 4, is_tokenized_sent: bool = False) -> ([int], str):
        sentences = text
        if not is_tokenized_sent:
            sentences = split_to_sentences(text)

        sentence_vectors = []

        for sentence in sentences:
            sentence_vectors.append(self._word_embedding.calculate_vector_avg(sentence))

        similarity_matrix = np.zeros([len(sentences), len(sentences)])

        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    similarity_matrix[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, Constants.WORD_EMBEDDING_DIMENSION), sentence_vectors[j].reshape(1, Constants.WORD_EMBEDDING_DIMENSION))[0,0]

        # Find out what happens here
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph, max_iter=1000)

        # Check how pagerank handles redundancy

        ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
        selected_sentences = []

        for i in range(n_sentence):
            selected_sentences.append(ranked_sentences[i][1])

        indices = []
        for selected_sentence in selected_sentences:
            indices.append(sentences.index(selected_sentence))

        indices.sort()

        summarize_result = ''
        for index in indices:
            summarize_result += ' ' + sentences[index]

        return indices, summarize_result.strip()

    # TODO: Find optimum max_length for each sentence
    def textrank_flatten(self, text, max_length: int = 0, n_sentence: int = 5, is_tokenized_sent: bool = False) -> ([int], str):
        sentences = text
        if not is_tokenized_sent:
            sentences = split_to_sentences(text)
        sentence_vectors = []
        max_sentence_length = 0

        for sentence in sentences:
            sentence_vectors.append(self._word_embedding.calculate_vector_flatten(sentence, 40))
            max_sentence_length = max(max_sentence_length, len(sentence.split()))

        if max_length == 0:
            max_length = max_sentence_length
            # DEBUG
            print(max_length)

        similarity_matrix = np.zeros([len(sentences), len(sentences)])

        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    similarity_matrix[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, Constants.WORD_EMBEDDING_DIMENSION * 40), sentence_vectors[j].reshape(1, Constants.WORD_EMBEDDING_DIMENSION * 40))[0, 0]

        # Find out what happens here
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph, max_iter=1000)

        # Check how pagerank handle redundancy

        ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
        selected_sentences = []

        for i in range(n_sentence):
            selected_sentences.append(ranked_sentences[i][1])

        indices = []
        for selected_sentence in selected_sentences:
            indices.append(sentences.index(selected_sentence))

        indices.sort()

        summarize_result = ''
        for index in indices:
            summarize_result += ' ' + sentences[index]

        return indices, summarize_result.strip()

    @staticmethod
    def n_first_sentences(text, n_sentence: int, is_tokenized_sent: bool = False) -> ([int], str):
        sentences = text
        if not is_tokenized_sent:
            sentences = split_to_sentences(text)

        indices = []
        summarize_result = ''

        for index in range(n_sentence):
            summarize_result += ' ' + sentences[index]
            indices.append(index)

        return indices, summarize_result.strip()
