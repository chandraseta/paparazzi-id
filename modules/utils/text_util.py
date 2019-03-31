def load_stopwords(stopwords_filepath: str) -> [str]:
    sw_file = open(stopwords_filepath)
    stopwords = []

    for line in sw_file:
        stopwords.append(line.strip())

    return stopwords