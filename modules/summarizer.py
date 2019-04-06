import gensim
import networkx as nx
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from modules.text_processor.sentence_splitter import split_to_sentences
from modules.utils.text_util import load_stopwords
from modules.word_embedding import WordEmbedding


class Summarizer:
    # TODO: Find out about multi-document summarization

    def __init__(self, word_embedding: WordEmbedding):
        self._word_embedding = word_embedding

    # TODO: Implement always get first sentence from text
    def textrank_avg(self, text: str, n_sentence: int = 5) -> str:
        sentences = split_to_sentences(text)
        sentence_vectors = []

        for sentence in sentences:
            sentence_vectors.append(self._calculate_sentence_vector(sentence))

        similarity_matrix = np.zeros([len(sentences), len(sentences)])

        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    similarity_matrix[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, self._word_embedding.dimension), sentence_vectors[j].reshape(1, self._word_embedding.dimension))[0,0]

        # Find out what happens here
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)

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

        return summarize_result.strip()

    # TODO: Find optimum max_length for each sentence
    def textrank_flatten(self, text: str, max_length: int = 0, n_sentence: int = 5) -> str:
        sentences = split_to_sentences(text)
        sentence_vectors = []
        max_sentence_length = 0

        for sentence in sentences:
            sentence_vectors.append(self._calculate_flattened_sentence_vector(sentence, 40))
            max_sentence_length = max(max_sentence_length, len(sentence.split()))

        if max_length == 0:
            max_length = max_sentence_length
            # DEBUG
            print(max_length)

        similarity_matrix = np.zeros([len(sentences), len(sentences)])

        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    similarity_matrix[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, self._word_embedding._dimension * 40), sentence_vectors[j].reshape(1, self._word_embedding._dimension * 40))[0,0]

        # Find out what happens here
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)

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

        return summarize_result.strip()

    def _calculate_sentence_vector(self, sentence: str, remove_stopwords=True) -> np.ndarray:
        sentence_tokens = gensim.utils.simple_preprocess(sentence)
        return self._calculate_sentence_vector(sentence_tokens, remove_stopwords)

    def _calculate_sentence_vector(self, sentence_tokens: [str], remove_stopwords: bool = True) -> np.ndarray:
        sentence_vector = np.zeros((self._dimension,))
        n_token = 0

        if remove_stopwords:
            stopwords = load_stopwords()

            for token in sentence_tokens:
                if len(token) != 0 and token not in stopwords:
                    sentence_vector += self.word2vec(token)
                    n_token += 1
        else:
            for token in sentence_tokens:
                if len(token) != 0:
                    sentence_vector += self.word2vec(token)
                    n_token += 1

        sentence_vector = sentence_vector / float(n_token)
        
        return sentence_vector

    def _calculate_flattened_sentence_vector(self, sentence: str, max_length: int = 30, remove_stopwords: bool = True) -> np.ndarray:
        sentence_tokens = gensim.utils.simple_preprocess(sentence)
        return self._calculate_flattened_sentence_vector(sentence_tokens, max_length, remove_stopwords)

    def _calculate_flattened_sentence_vector(self, sentence_tokens: [str], max_length: int = 30, remove_stopwords: bool = True) -> np.ndarray:
        sentence_vector = np.zeros(0)
        n_token = 0

        if remove_stopwords or len(sentence_tokens) > max_length:
            stopwords = load_stopwords()

            for token in sentence_tokens:
                if len(token) != 0 and n_token < max_length and token not in stopwords:
                    sentence_vector = np.append(sentence_vector, self._word_embedding.word2vec(token))
                    n_token += 1

        else:
            for token in sentence_tokens:
                if len(token) != 0 and n_token < max_length:
                    sentence_vector = np.append(sentence_vector, self._word_embedding.word2vec(token))
                    n_token += 1

        while n_token < max_length:
            sentence_vector = np.append(sentence_vector, np.zeros((self._word_embedding.dimension,)))
            n_token += 1

        return sentence_vector