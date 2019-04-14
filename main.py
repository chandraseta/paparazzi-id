from modules.sentence_embedding import SentenceEmbedding
from modules.summarizer.preprocessor import Preprocessor
from modules.summarizer.models import FFNNModel, GRUModel, LSTMModel
from modules.summarizer.models import BiGRUModel, BiLSTMModel


sentence_embedding = SentenceEmbedding()

train_sentences, train_labels = Preprocessor.load_indosum_data_by_sentence(sentence_embedding, type=['train.01', 'dev.01'])
train_seq_sentences, train_seq_labels = Preprocessor.load_indosum_data(sentence_embedding, type=['train.01', 'dev.01'])

test_sentences, test_labels = Preprocessor.load_indosum_data_by_sentence(sentence_embedding, type=['test.01'])
test_seq_sentences, test_seq_labels = Preprocessor.load_indosum_data(sentence_embedding, type=['test.01'])


def train_ffnn_model():
    ffnn = FFNNModel()
    ffnn.train(train_sentences, train_labels)
    ffnn.save('ffnn_v2')


def train_gru_model():
    gru = GRUModel()
    gru.train(train_seq_sentences, train_seq_labels)
    gru.save('gru_v2')


def train_lstm_model():
    lstm = LSTMModel()
    lstm.train(train_seq_sentences, train_seq_labels)
    lstm.save('lstm_v2')


def train_bigru_model():
    bigru = BiGRUModel()
    bigru.train(train_seq_sentences, train_seq_labels)
    bigru.save('bigru_v2')


def train_bilstm_model():
    bilstm = BiLSTMModel()
    bilstm.train(train_seq_sentences, train_seq_labels)
    bilstm.save('bilstm_v2')


def load_and_test_ffnn_model():
    ffnn = FFNNModel()
    ffnn.load('ffnn_v2')
    loss, accuracy, precision, recall, f1 = ffnn.evaluate(test_sentences, test_labels)

    logfile = open('models/ffnn.log', 'a+')
    logfile.write('Loss: {}\n'.format(loss))
    logfile.write('Accuracy: {}\n'.format(accuracy))
    logfile.write('Precision: {}'.format(precision))
    logfile.write('Recall: {}'.format(recall))
    logfile.write('F1: {}'.format(f1))


def load_and_test_gru_model():
    gru = GRUModel()
    gru.load('gru_v2')
    loss, accuracy, precision, recall, f1 = gru.evaluate(test_seq_sentences, test_seq_labels)

    logfile = open('models/gru.log', 'a+')
    logfile.write('Loss: {}\n'.format(loss))
    logfile.write('Accuracy: {}\n'.format(accuracy))
    logfile.write('Precision: {}'.format(precision))
    logfile.write('Recall: {}'.format(recall))
    logfile.write('F1: {}'.format(f1))


def load_and_test_lstm_model():
    lstm = LSTMModel()
    lstm.load('lstm_v2')
    loss, accuracy, precision, recall, f1 = lstm.evaluate(test_seq_sentences, test_seq_labels)

    logfile = open('models/lstm.log', 'a+')
    logfile.write('Loss: {}\n'.format(loss))
    logfile.write('Accuracy: {}\n'.format(accuracy))
    logfile.write('Precision: {}'.format(precision))
    logfile.write('Recall: {}'.format(recall))
    logfile.write('F1: {}'.format(f1))


def load_and_test_bigru_model():
    bigru = BiGRUModel()
    bigru.load('bigru_v2')
    loss, accuracy, precision, recall, f1 = bigru.evaluate(test_seq_sentences, test_seq_labels)

    logfile = open('models/bigru.log', 'a+')
    logfile.write('Loss: {}\n'.format(loss))
    logfile.write('Accuracy: {}\n'.format(accuracy))
    logfile.write('Precision: {}'.format(precision))
    logfile.write('Recall: {}'.format(recall))
    logfile.write('F1: {}'.format(f1))


def load_and_test_bilstm_model():
    bilstm = BiLSTMModel()
    bilstm.load('bilstm_v2')
    loss, accuracy, precision, recall, f1 = bilstm.evaluate(test_seq_sentences, test_seq_labels)

    logfile = open('models/bilstm.log', 'a+')
    logfile.write('Loss: {}\n'.format(loss))
    logfile.write('Accuracy: {}\n'.format(accuracy))
    logfile.write('Precision: {}'.format(precision))
    logfile.write('Recall: {}'.format(recall))
    logfile.write('F1: {}'.format(f1))


if __name__=='__main__':
    train_ffnn_model()
    train_gru_model()
    train_lstm_model()
    train_bigru_model()
    train_bilstm_model()

    load_and_test_ffnn_model()
    load_and_test_gru_model()
    load_and_test_lstm_model()
    load_and_test_bigru_model()
    load_and_test_bilstm_model()
