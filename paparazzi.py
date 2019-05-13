import argparse
import difflib
import os
import random

from modules.constants import Constants
from modules.scraper.detik import Detik
from modules.scraper.wikipedia import Wikipedia
from modules.word_embedding import WordEmbedding
from modules.summarizer.models import BiGRUModel
from modules.utils.scraped_data_util import get_scraped_data
from modules.utils.text_util import check_entity
from modules.section_assigner import SectionAssigner


def get_optimized_parameters(in_name: str, in_should_crawl: bool) -> (str, bool):
    available_character_names = [char_name for char_name in os.listdir(Constants.DETIK_DATA_FILEPATH)]
    name = in_name
    should_crawl = in_should_crawl

    if name not in available_character_names:
        close_matches = difflib.get_close_matches(name, available_character_names, n=3, cutoff=0.5)
        if len(close_matches) == 0:
            print('\nThere is no character yet with that name')
            should_crawl = True
            print('Crawl mode activated')
        else:
            print('\nFound {} close match(es)'.format(len(close_matches)))
            for idx, close_match in enumerate(close_matches):
                print('{idx}. {name}'.format(idx=idx + 1, name=close_match))
            print('Choose 0 to continue with the name "{}"'.format(name))

            index_choice = -1
            str_choice = input('Choice: ')

            while not (0 <= index_choice <= len(close_matches)):
                if str_choice.isdigit() and 0 <= int(str_choice) <= len(close_matches):
                    index_choice = int(str_choice)
                else:
                    str_choice = input('Invalid input. Choice: ')

            if index_choice == 0:
                should_crawl = True
                print('Crawl mode activated')
            else:
                name = close_matches[index_choice - 1]
                print('Character name is set to {}'.format(name))

    return name, should_crawl


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Paparazzi - Find the latest information about anyone on the news!'
    )

    # parser.add_argument('--target',
    #                     type=str,
    #                     required=True,
    #                     help='Name of target character')

    parser.add_argument('--crawl',
                        default=False,
                        action='store_true',
                        help='Crawl the web for the latest news')

    args = parser.parse_args()

    word_embedding = WordEmbedding()
    summarizer_model = BiGRUModel()
    summarizer_model.load('bigru_v3')

    # name = args.target
    should_crawl = args.crawl

    name = ''

    while name != 'quit':
        name = input('\n\nCharacter name: ')
        name, should_crawl = get_optimized_parameters(name, should_crawl)

        wiki_data = Wikipedia.get_article(name, should_crawl)

        if should_crawl:
            Detik.get_links([name])
            Detik.clean_links([name])
            Detik.crawl_links([name])

        scraped_data = get_scraped_data(name)

        # Debug
        scraped_data = random.sample(scraped_data, k=10)
        list_news_sentences = []

        for data in scraped_data:
            list_news_sentences.append(data['sentences'])

        list_summarized_data = []
        for news_sentences in list_news_sentences:
            list_summarized_data.append(summarizer_model.summarize(news_sentences, word_embedding))

        for index, summary in enumerate(list_summarized_data):
            if len(summary) > 0:
                if check_entity(summary, name):
                    section, dist = SectionAssigner.assign_section(summary, wiki_data, word_embedding)
                    print('---')
                    print(list_news_sentences[index])
                    print()
                    print(summary)
                    print()
                    print('Assigned Section: {}'.format(section))