import os
import re
import json
import requests

from bs4 import BeautifulSoup
from urllib.parse import quote

from modules.constants import Constants
from modules.utils.text_util import split_to_sentences


class Detik:
    LINK_DATA_FILEPATH = 'data/scraped/detik/'

    n_link_new = 0
    n_link_deleted = 0

    @staticmethod
    def generate_filepath(character_name: str) -> str:
        new_filepath = Detik.LINK_DATA_FILEPATH + character_name.lower()
        if not os.path.exists(new_filepath):
            os.mkdir(new_filepath)
            print('New directory created {}'.format(new_filepath))
        return new_filepath + '/'

    @staticmethod
    def generate_link_filename(character_name: str) -> str:
        return Detik.generate_filepath(character_name) + 'links.txt'

    @staticmethod
    def generate_crawled_link_filename(character_name: str) -> str:
        return Detik.generate_filepath(character_name) + 'crawled_links.txt'

    @staticmethod
    def generate_jsonl_filename(character_name: str) -> str:
        return Detik.generate_filepath(character_name) + 'data.jsonl'

    @staticmethod
    def get_links(character_names: [str], limit: int = 50):
        for character_name in character_names:
            links = []
            page = 1
            out_of_news = False
            print('[CRAWL] Begin crawling for {} from detik.com'.format(character_name))

            n_article = 0
            print('[CRAWL] Article count: 0', end='\r')
            while len(links) < limit and not out_of_news:
                url = 'https://www.detik.com/search/searchnews?query=' + quote(character_name) +'&page=' + str(page)
                request = requests.get(url)

                data = request.text
                soup = BeautifulSoup(data, 'html.parser')

                articles = soup.find_all('article',)

                if len(articles) > 0:
                    for article in articles:
                        links.append(article.find('a')['href'])
                        n_article += 1
                        print('[CRAWL] Article count: {}'.format(n_article), end='\r')
                        if len(links) >= limit:
                            break
                else:
                    out_of_news = True

                page += 1

            outfilepath = Detik.generate_link_filename(character_name)

            outfile = open(outfilepath, 'a+')
            for link in links:
                outfile.write(link + '\n')

            print('[CRAWL] Got {} articles for {}'.format(len(links), character_name))

    @staticmethod
    def clean_links(character_names: [str] = None):
        if character_names is None:
            character_names = [name for name in os.listdir(Detik.LINK_DATA_FILEPATH)]

        for character_name in character_names:
            linkfilepath = Detik.generate_link_filename(character_name)

            links = []
            linkfile = open(linkfilepath, 'r')
            for line in linkfile:
                if 'detik' in line:
                    # Remove articles from 20.detik.com (videos)
                    if '20.detik.com' not in line:
                        links.append(line)

            unique_link = set(links)

            linkfile.close()

            linkfile = open(linkfilepath, 'w+')
            for link in unique_link:
                if link != '\n':
                    linkfile.write(link)

            print('[CRAWL] Removed {} duplicate links from {}'.format(len(links) - len(unique_link), character_name))

    @staticmethod
    def crawl_links(character_names: [str] = None):
        if character_names is None:
            character_names = [name for name in os.listdir(Detik.LINK_DATA_FILEPATH)]

        for character_name in character_names:
            crawledlinkfilepath = Detik.generate_crawled_link_filename(character_name)
            linkfilepath = Detik.generate_link_filename(character_name)
            datafilepath = Detik.generate_jsonl_filename(character_name)

            crawled_links = []

            if os.path.isfile(crawledlinkfilepath):
                crawledlinkfile = open(crawledlinkfilepath, 'r')
                for line in crawledlinkfile:
                    crawled_links.append(line)
                crawledlinkfile.close()
            else:
                crawledlinkfile = open(crawledlinkfilepath, 'a+')
                crawledlinkfile.close()

            links = []
            linkfile = open(linkfilepath, 'r')
            for line in linkfile:
                links.append(line)

            uncrawled_links = [link for link in links if link not in crawled_links]

            crawledlinkfile = open(crawledlinkfilepath, 'a+')
            datafile = open(datafilepath, 'a+')

            print('[CRAWL] Processing data for {}'.format(character_name), end='\r')

            for url in uncrawled_links:
                dict = {}
                dict['url'] = url.strip()

                request = requests.get(url)

                data = request.text
                soup = BeautifulSoup(data, 'html.parser')

                titles = soup.find_all('title')
                body_text = soup.find_all('div', class_='itp_bodycontent')

                if len(titles) > 0:
                    dict['title'] = titles[0].text
                else:
                    print('\tNo title found in: {}'.format(url))
                    dict['title'] = '-'

                sentences = []

                for text in body_text:

                    for br in text.find_all('br'):
                        br.replace_with(' ')

                    str_text = text.text
                    str_text = str_text.replace('\n', ' ')
                    str_text = re.sub('\s+', ' ', str_text).strip()

                    tokenized_sent = split_to_sentences(str_text)
                    for idx, sentence in enumerate(tokenized_sent):
                        if idx == 0:
                            if sentence.find('-') != -1 and sentence.find('-') < len(sentence) - 1:
                                sentence = sentence[sentence.find('-') + 1:]

                        if not ('baca juga' in sentence.lower() or 'simak juga' in sentence.lower()):
                            sentences.append(sentence.strip())

                dict['sentences'] = sentences

                crawledlinkfile.write(url)
                datafile.write(json.dumps(dict) + '\n')

            print('[CRAWL] Finished processing data for {}'.format(character_name))


if __name__ == '__main__':
    names = []

    namefile = open(Constants.CHARACTER_LIST_FILEPATH, 'r')

    for line in namefile:
        if line != '\n':
            names.append(line.strip('\n'))

    Detik.get_links(names)
    Detik.clean_links()
    Detik.crawl_links()
