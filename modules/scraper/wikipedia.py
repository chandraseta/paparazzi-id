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

        result = {}

        # Check empty pages
        pars = soup.find_all('p')
        if len(pars) == 0:
            return result
        else:
            # Remove tables
            for td in soup.find_all('td'):
                td.decompose()

            filtered_items = soup.find_all(['h2', 'p'])

            current_section = Constants.INITIAL_WIKI_SECTION
            current_section_content = []

            for filtered_item in filtered_items:
                if '<h2>' in str(filtered_item)[:4]:
                    if filtered_item.find('span', {'class': 'mw-headline'}):
                        # Skip empty sections
                        if len(current_section_content) > 0:
                            result[current_section] = current_section_content
                            current_section_content = []
                        current_section = filtered_item.text
                elif '<p>' in str(filtered_item)[:3]:
                    content = Wikipedia.clean_wikipedia_text(filtered_item.text)
                    current_section_content.append(content)

            if current_section not in result and len(current_section_content) > 0:
                result[current_section] = current_section_content

        return result

