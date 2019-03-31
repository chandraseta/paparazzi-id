import gensim
import networkx as nx
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from modules.text_processor.sentence_splitter import split_to_sentences
from modules.utils.text_util import load_stopwords
from modules.word_embedding import WordEmbedding


class Summarizer:
    def __init__(self, word_embedding: WordEmbedding):
        self._word_embedding = word_embedding

    # TODO: Implement always get first sentence from text
    def summarize(self, text, ratio: float = 0.25, n_sentence: int = 0) -> str:
        sentences = split_to_sentences(text)
        sentence_vectors = []

        for sentence in sentences:
            sentence_tokens = gensim.utils.simple_preprocess(sentence)
            vector = np.zeros((WE_DIM,))
            n_token = 0

            stopwords = load_stopwords('data/indo_stopwords.txt')

            for token in sentence_tokens:
                if len(token) != 0 and token not in stopwords:
                    vector += self._word_embedding.word2vec(token)
                    n_token += 1

            vector = vector / float(n_token)
            sentence_vectors.append(vector)

        similarity_matrix = np.zeros([len(sentences), len(sentences)])

        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    similarity_matrix[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,WE_DIM), sentence_vectors[j].reshape(1,WE_DIM))[0,0]

        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)

        ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
        selected_sentences = []

        if n_sentence > 0:
            n_sentence = max(n_sentence, len(sentences))
            for i in range(n_sentence):
                selected_sentences.append(ranked_sentences[i][1])
        else:
            for i in range(int(len(sentences) * ratio)):
                selected_sentences.append(ranked_sentences[i][1])

        indices = []
        for selected_sentence in selected_sentences:
            indices.append(sentences.index(selected_sentence))

        indices.sort()

        summarize_result = ''
        for index in indices:
            summarize_result += ' ' + sentences[index]

        return summarize_result.strip()