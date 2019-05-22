from sklearn.metrics.pairwise import cosine_similarity

from modules.constants import Constants
from modules.word_embedding import WordEmbedding


class SimilarityChecker:
    @staticmethod
    def calculate_sentence_similarity(sentence_1: str, sentence_2: str, word_embedding: WordEmbedding):
        sentence_vector_1 = word_embedding.calculate_vector_avg(sentence_1)
        sentence_vector_2 = word_embedding.calculate_vector_avg(sentence_2)

        similarity = cosine_similarity([sentence_vector_1], [sentence_vector_2])
        return similarity

    @staticmethod
    def calculate_paragraph_similarity(paragraph_1: [str], paragraph_2: [str], word_embedding: WordEmbedding):
        paragraph_vector_1 = word_embedding.calculate_paragraph_vector_avg(paragraph_1)
        paragraph_vector_2 = word_embedding.calculate_paragraph_vector_avg(paragraph_2)

        similarity = cosine_similarity([paragraph_vector_1], [paragraph_vector_2])
        return similarity

    @staticmethod
    def get_similar_summaries_indices_pair(list_summaries: [[str]], word_embedding: WordEmbedding) -> [dict]:
        similar_pair = []

        max_index = len(list_summaries)
        for index, checked_summary in enumerate(list_summaries):
            for j in range(index + 1, max_index):
                similarity = SimilarityChecker.calculate_paragraph_similarity(checked_summary, list_summaries[j], word_embedding)

                if similarity > Constants.SIMILARITY_THRESHOLD:
                    similar_pair.append({
                        'summary_1': checked_summary,
                        'index_1': index,
                        'summary_2': list_summaries[j],
                        'index_2': j,
                        'similarity': similarity
                    })

        return similar_pair
