STOPWORDS_ID_FILEPATH = 'data/indo_stopwords.txt'

def load_stopwords(stopwords_filepath: str = STOPWORDS_ID_FILEPATH) -> [str]:
    sw_file = open(stopwords_filepath)
    stopwords = []

    for line in sw_file:
        stopwords.append(line.strip())

    return stopwords