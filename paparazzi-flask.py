import argparse
import difflib
import os
import random

from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash

from modules.constants import Constants
from modules.scraper.detik import Detik
from modules.scraper.wikipedia import Wikipedia
from modules.summarizer.models import BiGRUModel
from modules.section_assigner import SectionAssigner
from modules.similarity_checker import SimilarityChecker
from modules.word_embedding import WordEmbedding
from modules.utils.scraped_data_util import get_scraped_data
from modules.utils.text_util import check_entity, join_sentences
from paparazzi import get_optimized_parameters


app = Flask(__name__)
app.config.from_object(__name__)

app.config.update(dict (
  SECRET_KEY = 'RC',
  USERNAME = 'admin',
  PASSWORD = 'admin'
))

def check_character_name(in_name: str) -> bool:
    available_character_names = [char_name for char_name in os.listdir(Constants.DETIK_DATA_FILEPATH)]
    return in_name in available_character_names


@app.route('/')
def showPage():
    global word_embedding
    global summarizer_model

    try:
        word_embedding
    except NameError:
        word_embedding = WordEmbedding()

    try:
        summarizer_model
    except NameError:
        summarizer_model = BiGRUModel()
        summarizer_model.load('bigru_v3')
        summarizer_model._model._make_predict_function()

    return render_template('page.html')


@app.route('/submit', methods=['POST'])
def submit():
    form = request.form
    name = form['input_name']
    should_crawl = False

    error_message = ''

    if not check_character_name(name):
        # Detik.get_links([name], limit=10)
        # Detik.clean_links([name])
        # Detik.crawl_links([name])
        error_message = 'Tidak ditemukan data untuk {}'.format(name)

    if name == '':
        error_message = 'Tidak ada masukan nama.'

    list_news_sentences = []
    list_failed_no_entity = []
    list_failed_too_similar = []
    list_candidate_paragraph_details = []

    if error_message == '':
        wiki_data = Wikipedia.get_article(name, should_crawl)
        scraped_data = get_scraped_data(name)

        scraped_data = random.sample(scraped_data, k=10)

        for data in scraped_data:
            list_news_sentences.append(data['sentences'])

        list_summarized_data = []
        for news_sentences in list_news_sentences:
            list_summarized_data.append(summarizer_model.summarize(news_sentences, word_embedding))

        list_duplicate_data_indices = []
        for index, summary in enumerate(list_summarized_data):
            if len(summary) > 0:
                summary_string = join_sentences(summary)

                if index not in list_duplicate_data_indices:
                    if check_entity(summary, name):
                        duplicate_data_indices = SimilarityChecker.get_similar_summaries_indices(
                            index, summary, list_summarized_data, word_embedding
                        )

                        if len(duplicate_data_indices) > 0:
                            for ind in duplicate_data_indices:
                                list_duplicate_data_indices.append(ind)

                        section_recommendation, sim = SectionAssigner.assign_section_cosine(summary, wiki_data, word_embedding)

                        candidate_paragraph_detail = {
                            'summary': summary_string,
                            'section': section_recommendation,
                            'news': list_news_sentences[index]
                        }
                        list_candidate_paragraph_details.append(candidate_paragraph_detail)
                    else:
                        list_failed_no_entity.append(summary_string)
                else:
                    list_failed_too_similar.append(summary_string)

    return render_template('page.html', input=name, wiki_link=Wikipedia.convert_to_wiki_url(name), error=error_message,
                           result_list=list_candidate_paragraph_details, no_entity_list=list_failed_no_entity,
                           too_similar_list=list_failed_too_similar)
