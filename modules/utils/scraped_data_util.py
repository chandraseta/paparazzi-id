import os
import json

from modules.constants import Constants


def generate_filepath(character_name: str) -> str:
    new_filepath = Constants.DETIK_DATA_FILEPATH + character_name
    if not os.path.exists(new_filepath):
        os.mkdir(new_filepath)
        print('[UTIL] New directory created {}'.format(new_filepath))
    return new_filepath + '/'


def generate_link_filename(character_name: str) -> str:
    return generate_filepath(character_name) + 'links.txt'


def generate_crawled_link_filename(character_name: str) -> str:
    return generate_filepath(character_name) + 'crawled_links.txt'


def generate_data_filename(character_name: str) -> str:
    return generate_filepath(character_name) + 'data.jsonl'


def generate_used_data_filename(character_name: str) -> str:
    return generate_filepath(character_name) + 'used_data.jsonl'


def get_scraped_data(name: str, include_used_data: bool = False):
    print('[UTIL] Getting news articles for {}'.format(name))
    data_filepath = generate_data_filename(name)
    used_data_filepath = generate_data_filename(name)

    if not os.path.isfile(used_data_filepath):
        used_data_file = open(used_data_filepath, 'a+')
        used_data_file.close()

    list_data_dict = []

    if include_used_data:
        used_data_file = open(used_data_filepath, 'r')
        for line in used_data_file:
            data = json.loads(line)
            list_data_dict.append(data)

    data_file = open(data_filepath, 'r')
    for line in data_file:
        data = json.loads(line)
        list_data_dict.append(data)

    return list_data_dict


def write_back_data(name: str, unused_data: [dict], used_data: [dict]):
    data_filepath = generate_data_filename(name)
    used_data_filepath = generate_data_filename(name)

    data_file = open(data_filepath, 'a+')

    for data_dict in unused_data:
        data_file.write(json.dumps(data_dict) + '\n')

    used_data_file = open(used_data_filepath, 'a+')

    for data_dict in used_data:
        used_data_file.write(json.dumps(data_dict) + '\n')

    print('[UTIL] Data for {} written to {} and {}'.format(name, data_filepath, used_data_filepath))
