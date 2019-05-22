from modules.summarizer.models import BiGRUModel
from modules.word_embedding import WordEmbedding
from modules.utils.evaluator import calculate_average_rouge_score
from modules.utils.indosum_util import get_articles_summaries_indices
from modules.utils.text_util import join_sentences

word_embedding = WordEmbedding()
summarizer_model = BiGRUModel()
summarizer_model.load('bigru_v3')

articles, summaries, _ = get_articles_summaries_indices(data_types=['test.01'])

total_rouge_1_f = 0
total_rouge_1_p = 0
total_rouge_1_r = 0

total_rouge_2_f = 0
total_rouge_2_p = 0
total_rouge_2_r = 0

total_rouge_l_f = 0
total_rouge_l_p = 0
total_rouge_l_r = 0

for index, article in enumerate(articles):
    prediction = summarizer_model.summarize(article, word_embedding)
    joined_prediction = join_sentences(prediction)

    rouge_score = calculate_average_rouge_score([joined_prediction], [summaries[index]])

    total_rouge_1_f += rouge_score['rouge-1']['f']
    total_rouge_1_p += rouge_score['rouge-1']['p']
    total_rouge_1_r += rouge_score['rouge-1']['r']

    total_rouge_2_f += rouge_score['rouge-2']['f']
    total_rouge_2_p += rouge_score['rouge-2']['p']
    total_rouge_2_r += rouge_score['rouge-2']['r']

    total_rouge_l_f += rouge_score['rouge-l']['f']
    total_rouge_l_p += rouge_score['rouge-l']['f']
    total_rouge_l_r += rouge_score['rouge-l']['f']

n_article = len(articles)

outfile = open('rouge.txt', 'w+')
outfile.write('rouge-1[f]: {} \n'.format(total_rouge_1_f / n_article))
outfile.write('rouge-1[p]: {} \n'.format(total_rouge_1_p / n_article))
outfile.write('rouge-1[r]: {} \n'.format(total_rouge_1_r / n_article))
outfile.write('rouge-2[f]: {} \n'.format(total_rouge_2_f / n_article))
outfile.write('rouge-2[p]: {} \n'.format(total_rouge_2_p / n_article))
outfile.write('rouge-2[r]: {} \n'.format(total_rouge_2_r / n_article))
outfile.write('rouge-l[f]: {} \n'.format(total_rouge_l_f / n_article))
outfile.write('rouge-l[p]: {} \n'.format(total_rouge_l_p / n_article))
outfile.write('rouge-l[r]: {} \n'.format(total_rouge_l_r / n_article))
