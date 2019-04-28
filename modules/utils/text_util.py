import re

from modules.constants import Constants


def load_stopwords(stopwords_filepath: str = Constants.STOPWORDS_ID_FILEPATH) -> [str]:
    sw_file = open(stopwords_filepath)
    stopwords = []

    for line in sw_file:
        stopwords.append(line.strip())

    return stopwords


def split_to_sentences(text: str) -> [str]:
    # Convert . -> $$ inside parenthesis and quotes
    text = re.sub('\.(?=[^(]*\\))', '$$', text)
    text = re.sub(r"(?<=([\"]\b))(?:(?=(\\?))\2.)*?(?=\1)", lambda x:x.group(0).replace('.', '$$'), text)

    sentences = text.split('. ')
    temp_sentence = ''
    return_sentences = []

    print('FirstSent: {}'.format(sentences))

    for idx, sentence in enumerate(sentences):
        sentence = sentence.strip()

        if sentence != '':
            if temp_sentence == '':
                temp_sentence = sentence
            else:
                temp_sentence += '. ' + sentence

            # Check if the first word in next sentence begins with uppercase letter
            next_sentence = '\\'
            if idx+1 < len(sentences):
                if sentences[idx+1] != '':
                    next_sentence = sentences[idx+1]

            # Also check if last word is less than 3 character
            if len(sentence.split(' ')[-1]) < 3 or next_sentence == '\\' or next_sentence[0] != next_sentence[0].upper():
                continue
            else:
                # Convert back $$ -> .
                temp_sentence = temp_sentence.replace('$$', '.')

                # Add period
                temp_sentence += '.'

                # Export sentence
                if len(temp_sentence) >= 20:

                    # Remove multiple periods
                    temp_sentence = re.sub('\.+$', '.', temp_sentence)

                    # Remove multiple whitespaces
                    temp_sentence = re.sub('\s+', ' ', temp_sentence).strip()
                    return_sentences.append(temp_sentence)

                temp_sentence = ''

    if len(temp_sentence) >= 20:
        # Remove multiple periods
        temp_sentence = re.sub('\.+$', '.', temp_sentence)

        # Remove multiple whitespaces
        temp_sentence = re.sub('\s+', ' ', temp_sentence).strip()
        return_sentences.append(temp_sentence)

    return return_sentences
