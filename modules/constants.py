class Constants:
    DETIK_DATA_FILEPATH = 'data/scraped/detik/'
    WIKIPEDIA_DATA_FILEPATH = 'data/scraped/wikipedia/'
    CHARACTER_LIST_FILEPATH = 'data/scraped/name_list.txt'

    SCRAPED_DATA_URL = 'url'
    SCRAPED_DATA_TITLE = 'title'
    SCRAPED_DATA_SENTENCES = 'sentences'

    INDOSUM_FILEPATH = 'data/indosum/'
    INDOSUM_EXTENSION = '.jsonl'

    HTML_EXTENSION = '.html'

    INITIAL_WIKI_SECTION = 'Lead paragraph'

    MAXIMUM_SENTENCE_LENGTH = 35
    MODEL_PATH = 'models/'
    MODEL_THRESHOLD = 0.3

    NO_LEADING_SPACE_SYMBOLS = [',', '.', '!', '?']

    STOPWORDS_ID_FILEPATH = 'data/indo_stopwords.txt'

    WORD_EMBEDDING_FILEPATH = 'w2v/indonesian.300d.100.txt'
    WORD_EMBEDDING_DIMENSION = 300
