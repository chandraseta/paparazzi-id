from rouge import Rouge

from modules.utils.text_util import join_sentences


def calculate_average_rouge_score(predictions: [[str]], gold_summaries: [[str]]) -> dict:
    joined_predictions = [join_sentences(x) for x in predictions]
    joined_gold_summaries = [join_sentences(x) for x in gold_summaries]

    rouge = Rouge()
    rouge_score = rouge.get_scores(joined_predictions, joined_gold_summaries, avg=True)

    return rouge_score
