import os
import re
import requests

from bs4 import BeautifulSoup

from modules.constants import Constants


class Wikipedia:
    @staticmethod
    def convert_to_wiki_url(character_name: str) -> [[str]]:
        return character_name.replace(' ', '_')

    @staticmethod
    def clean_wikipedia_text(text: str) -> str:
        text = re.sub("[\[].*?[\]]", "", text)
        return text

    @staticmethod
    def get_article(character_name: str, crawl_mode: False) -> dict:
        print('[CRAWL] Fetching article for {} from Wikipedia'.format(character_name))
        data = None
        if not crawl_mode:
            wikifile_path = Constants.WIKIPEDIA_DATA_FILEPATH + character_name + Constants.HTML_EXTENSION
            if os.path.isfile(wikifile_path):
                wikifile = open(wikifile_path, 'r')
                data = wikifile.read()

        if data is None:
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

            filtered_items = soup.find_all(['h2', 'p', 'li'])

            current_section = Constants.INITIAL_WIKI_SECTION
            current_section_content = ''

            for filtered_item in filtered_items:
                if '<h2>' in str(filtered_item)[:4]:
                    if filtered_item.find('span', {'class': 'mw-headline'}):
                        # Skip empty sections
                        if len(current_section_content) > 0 \
                                and current_section.lower() != 'referensi' \
                                and current_section.lower() != 'pranala luar'\
                                and current_section.lower() != 'lihat pula':
                            current_section_content = re.sub('\s+', ' ', current_section_content).strip()
                            result[current_section] = current_section_content
                            current_section_content = ''
                        current_section = Wikipedia.clean_wikipedia_text(filtered_item.text)
                elif '<p>' in str(filtered_item)[:3] or '<li>' in str(filtered_item)[:4]:
                    content = Wikipedia.clean_wikipedia_text(filtered_item.text)
                    current_section_content += content + ' '

            if current_section not in result \
                    and len(current_section_content) > 0 \
                    and current_section.lower() != 'referensi' \
                    and current_section.lower() != 'pranala luar' \
                    and current_section.lower() != 'lihat pula':
                current_section_content = re.sub('\s+', ' ', current_section_content).strip()
                result[current_section] = current_section_content

        return result

    @staticmethod
    def save_article(character_name: str):
        print('[CRAWL] Saving article for {} from Wikipedia'.format(character_name))

        url = 'https://id.wikipedia.org/wiki/' + Wikipedia.convert_to_wiki_url(character_name)
        request = requests.get(url)
        data = request.text

        outfile_path = Constants.WIKIPEDIA_DATA_FILEPATH + character_name + Constants.HTML_EXTENSION
        outfile = open(outfile_path, 'w+')
        outfile.write(data)

        print('[CRAWL] Wikipedia data saved to {}'.format(outfile_path))
