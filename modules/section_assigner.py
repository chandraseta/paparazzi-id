from sklearn.metrics.pairwise import pairwise_distances

from modules.word_embedding import WordEmbedding
from modules.utils.text_util import split_to_sentences
from modules.similarity_checker import SimilarityChecker


class SectionAssigner:
    @staticmethod
    def assign_section(sentences: [str], wikipedia: dict, word_embedding: WordEmbedding) -> (str, int):
        nearest_section = None
        nearest_distance = None

        text = ''
        for sentence in sentences:
            text += sentence + ' '

        text = text.strip()
        sentence_vector = word_embedding.calculate_vector_avg(text)

        centroids = {}
        for section, content in wikipedia.items():
            paragraph = split_to_sentences(content)
            centroids[section] = word_embedding.calculate_paragraph_vector_avg(paragraph)

        for section, vector in centroids.items():
            if nearest_section is None or nearest_distance is None:
                nearest_section = section
                nearest_distance = pairwise_distances([vector], [sentence_vector], metric='euclidean')
            else:
                current_distance = pairwise_distances([vector], [sentence_vector], metric='euclidean')
                if current_distance < nearest_distance:
                    nearest_section = section
                    nearest_distance = current_distance

        return nearest_section, nearest_distance

    @staticmethod
    def assign_section_cosine(sentences: [str], wikipedia: dict, word_embedding: WordEmbedding) -> (str, int):
        most_similar_section = None
        highest_similarity = None

        for section, content in wikipedia.items():
            wiki_paragraph = split_to_sentences(content)

            if most_similar_section is None or highest_similarity is None:
                most_similar_section = section
                highest_similarity = SimilarityChecker.calculate_paragraph_similarity(
                    sentences, wiki_paragraph, word_embedding
                )
            else:
                current_similarity = SimilarityChecker.calculate_paragraph_similarity(
                    sentences, wiki_paragraph, word_embedding
                )
                if current_similarity > highest_similarity:
                    most_similar_section = section
                    highest_similarity = current_similarity

        return most_similar_section, highest_similarity
