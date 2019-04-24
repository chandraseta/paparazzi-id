import os
import re
import requests

from bs4 import BeautifulSoup

from modules.constants import Constants
from modules.utils.text_util import split_to_sentences


class Wikipedia:
    @staticmethod
    def convert_to_wiki_url(character_name: str) -> [[str]]:
        return character_name.title().replace(' ', '_')

    @staticmethod
    def clean_wikipedia_text(text: str) -> str:
        text = re.sub("[\[].*?[\]]", "", text)
        return text

    @staticmethod
    def get_article(character_name: str) -> [str]:
        print('[CRAWL] Fetching article for {} from Wikipedia'.format(character_name.title()))

        url = 'https://id.wikipedia.org/wiki/' + Wikipedia.convert_to_wiki_url(character_name)
        request = requests.get(url)

        data = request.text
        soup = BeautifulSoup(data, 'html.parser')

        # Remove tables
        for td in soup.find_all('td'):
            td.decompose()

        filtered_items = soup.find_all(['h2', 'p'])

        for filtered_item in filtered_items:
            if '<h2>' in str(filtered_item)[:4]:
                if filtered_item.find('span', {'class': 'mw-headline'}):
                    print('Aye')
                    print(filtered_item.text)
            elif '<p>' in str(filtered_item)[:3]:
                print(filtered_item.text)



if __name__ == '__main__':
    Wikipedia.get_article('joko widodo')
