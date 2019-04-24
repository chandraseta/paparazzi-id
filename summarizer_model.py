import argparse
import numpy as np

from modules.sentence_embedding import SentenceEmbedding
from modules.summarizer.preprocessor import Preprocessor
from modules.summarizer.models import FFNNModel, GRUModel, LSTMModel
from modules.summarizer.models import BiGRUModel, BiLSTMModel


def train_ffnn_model(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, version: str):
    model_name = 'ffnn_' + version
    ffnn = FFNNModel()
    ffnn.train(x_train, y_train, x_val, y_val, model_name + '_hist.pickle')
    ffnn.save(model_name)


def train_gru_model(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, version: str):
    model_name = 'gru_' + version
    gru = GRUModel()
    gru.train(x_train, y_train, x_val, y_val, model_name + '_hist.pickle')
    gru.save('gru_' + version)


def train_lstm_model(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, version: str):
    model_name = 'lstm_' + version
    lstm = LSTMModel()
    lstm.train(x_train, y_train, x_val, y_val, model_name + '_hist.pickle')
    lstm.save('lstm_' + version)


def train_bigru_model(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, version: str):
    model_name = 'bigru_' + version
    bigru = BiGRUModel()
    bigru.train(x_train, y_train, x_val, y_val, model_name + '_hist.pickle')
    bigru.save('bigru_' + version)


def train_bilstm_model(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, version: str):
    model_name = 'bilstm_' + version
    bilstm = BiLSTMModel()
    bilstm.train(x_train, y_train, x_val, y_val, model_name + '_hist.pickle')
    bilstm.save('bilstm_' + version)


def load_and_test_ffnn_model(x_test: np.ndarray, y_test: np.ndarray, version: str):
    model_name = 'ffnn_' + version
    ffnn = FFNNModel()
    ffnn.load(model_name)
    loss, accuracy, precision, recall, f1 = ffnn.evaluate(x_test, y_test)
    print('Loss: {} Accuracy: {} Precision: {} Recall: {} F1: {}'.format(loss, accuracy, precision, recall, f1))

    logfile = open('models/ffnn.log', 'a+')
    logfile.write('Version: {}\n'.format(version))
    logfile.write('Loss: {}\n'.format(loss))
    logfile.write('Accuracy: {}\n'.format(accuracy))
    logfile.write('Precision: {}\n'.format(precision))
    logfile.write('Recall: {}\n'.format(recall))
    logfile.write('F1: {}\n\n'.format(f1))


def load_and_test_gru_model(x_test: np.ndarray, y_test: np.ndarray, version: str):
    model_name = 'gru_' + version
    gru = GRUModel()
    gru.load(model_name)
    loss, accuracy, precision, recall, f1 = gru.evaluate(x_test, y_test)
    print('Loss: {} Accuracy: {} Precision: {} Recall: {} F1: {}'.format(loss, accuracy, precision, recall, f1))

    logfile = open('models/gru.log', 'a+')
    logfile.write('Version: {}\n'.format(version))
    logfile.write('Loss: {}\n'.format(loss))
    logfile.write('Accuracy: {}\n'.format(accuracy))
    logfile.write('Precision: {}\n'.format(precision))
    logfile.write('Recall: {}\n'.format(recall))
    logfile.write('F1: {}\n\n'.format(f1))


def load_and_test_lstm_model(x_test: np.ndarray, y_test: np.ndarray, version: str):
    model_name = 'lstm_' + version
    lstm = LSTMModel()
    lstm.load(model_name)
    loss, accuracy, precision, recall, f1 = lstm.evaluate(x_test, y_test)
    print('Loss: {} Accuracy: {} Precision: {} Recall: {} F1: {}'.format(loss, accuracy, precision, recall, f1))

    logfile = open('models/lstm.log', 'a+')
    logfile.write('Version: {}\n'.format(version))
    logfile.write('Loss: {}\n'.format(loss))
    logfile.write('Accuracy: {}\n'.format(accuracy))
    logfile.write('Precision: {}\n'.format(precision))
    logfile.write('Recall: {}\n'.format(recall))
    logfile.write('F1: {}\n\n'.format(f1))


def load_and_test_bigru_model(x_test: np.ndarray, y_test: np.ndarray, version: str):
    model_name = 'bigru_' + version
    bigru = BiGRUModel()
    bigru.load(model_name)
    loss, accuracy, precision, recall, f1 = bigru.evaluate(x_test, y_test)
    print('Loss: {} Accuracy: {} Precision: {} Recall: {} F1: {}'.format(loss, accuracy, precision, recall, f1))

    logfile = open('models/bigru.log', 'a+')
    logfile.write('Version: {}\n'.format(version))
    logfile.write('Loss: {}\n'.format(loss))
    logfile.write('Accuracy: {}\n'.format(accuracy))
    logfile.write('Precision: {}\n'.format(precision))
    logfile.write('Recall: {}\n'.format(recall))
    logfile.write('F1: {}\n\n'.format(f1))


def load_and_test_bilstm_model(x_test: np.ndarray, y_test: np.ndarray, version: str):
    model_name = 'bilstm_' + version
    bilstm = BiLSTMModel()
    bilstm.load(model_name)
    loss, accuracy, precision, recall, f1 = bilstm.evaluate(x_test, y_test)
    print('Loss: {} Accuracy: {} Precision: {} Recall: {} F1: {}'.format(loss, accuracy, precision, recall, f1))

    logfile = open('models/bilstm.log', 'a+')
    logfile.write('Version: {}\n'.format(version))
    logfile.write('Loss: {}\n'.format(loss))
    logfile.write('Accuracy: {}\n'.format(accuracy))
    logfile.write('Precision: {}\n'.format(precision))
    logfile.write('Recall: {}\n'.format(recall))
    logfile.write('F1: {}\n\n'.format(f1))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train summarizer models')

    parser.add_argument('--version',
                        type=str,
                        required=True,
                        help='Model version')

    parser.add_argument('--data',
                        type=str,
                        choices=['01', '02', '03', '04', '05'],
                        required=True,
                        help='Dataset choice')

    parser.add_argument('--train',
                        default=False,
                        action='store_true',
                        help='Training mode')

    parser.add_argument('--full_train',
                        default=False,
                        action='store_true',
                        help='Do full training on all data')

    parser.add_argument('--test',
                        default=False,
                        action='store_true',
                        help='Testing mode')

    parser.add_argument('--ffnn',
                        default=False,
                        action='store_true',
                        help='Use FFNN model')

    parser.add_argument('--gru',
                        default=False,
                        action='store_true',
                        help='Use GRU model')

    parser.add_argument('--lstm',
                        default=False,
                        action='store_true',
                        help='Use LSTM model')

    parser.add_argument('--bigru',
                        default=False,
                        action='store_true',
                        help='Use BiGRU model')

    parser.add_argument('--bilstm',
                        default=False,
                        action='store_true',
                        help='Use BiLSTM model')

    args = parser.parse_args()
    any_mode_selected = args.train or args.test or args.full_train
    any_model_selected = args.ffnn or args.gru or args.lstm or args.bigru or args.bilstm

    if any_mode_selected and any_model_selected:
        sentence_embedding = SentenceEmbedding()

        if args.train:
            train_data_type = ['train.' + args.data]
            val_data_type = ['dev.' + args.data]

            if args.ffnn:
                print('[TRAIN] Loading data for dataset {}'.format(args.data))
                train_sentences, train_labels = Preprocessor.load_indosum_data_by_sentence(sentence_embedding, type=train_data_type)
                val_sentences, val_labels = Preprocessor.load_indosum_data_by_sentence(sentence_embedding, type=val_data_type)
                print('[TRAIN] Finished loading data')

                print('[TRAIN] FFNN Model')
                train_ffnn_model(train_sentences, train_labels, val_sentences, val_labels, args.version)

            if args.gru or args.lstm or args.bigru or args.bilstm:
                print('[TRAIN] Loading sequence data for dataset {}'.format(args.data))
                train_seq_sentences, train_seq_labels = Preprocessor.load_indosum_data(sentence_embedding, type=train_data_type)
                val_seq_sentences, val_seq_labels = Preprocessor.load_indosum_data(sentence_embedding, type=val_data_type)
                print('[TRAIN] Finished loading sequence data')

                if args.gru:
                    print('[TRAIN] GRU Model')
                    train_gru_model(train_seq_sentences, train_seq_labels, val_seq_sentences, val_seq_labels, args.version)

                if args.lstm:
                    print('[TRAIN] LSTM Model')
                    train_lstm_model(train_seq_sentences, train_seq_labels, val_seq_sentences, val_seq_labels, args.version)

                if args.bigru:
                    print('[TRAIN] BiGRU Model')
                    train_bigru_model(train_seq_sentences, train_seq_labels, val_seq_sentences, val_seq_labels, args.version)

                if args.bilstm:
                    print('[TRAIN] BiLSTM Model')
                    train_bilstm_model(train_seq_sentences, train_seq_labels, val_seq_sentences, val_seq_labels, args.version)

        if args.full_train:
            full_train_data_type = ['train.' + args.data, 'dev.' + args.data, 'test.' + args.data]

            if args.ffnn:
                print('[FULL TRAIN] Loading data for dataset {}'.format(args.data))
                full_train_sentences, full_train_labels = Preprocessor.load_indosum_data_by_sentence(sentence_embedding, type=full_train_data_type)
                print('[FULL TRAIN] Finished loading data')

            if args.gru or args.lstm or args.bigru or args.bilstm:
                print('[FULL TRAIN] Loading sequence data for dataset {}'.format(args.data))
                full_train_seq_sentences, full_train_seq_labels = Preprocessor.load_indosum_data(sentence_embedding, type=full_train_data_type)
                dev_seq_sentences, dev_seq_labels = Preprocessor.load_indosum_data(sentence_embedding, type=['dev' + args.data])
                print('[FULL TRAIN] Finished loading sequence data')

                if args.gru:
                    print('[FULL TRAIN] GRU Model')
                    train_gru_model(full_train_seq_sentences, full_train_seq_labels, dev_seq_sentences, dev_seq_labels, args.version)

                if args.lstm:
                    print('[FULL TRAIN] LSTM Model')
                    train_lstm_model(full_train_seq_sentences, full_train_seq_labels, dev_seq_sentences, dev_seq_labels, args.version)

                if args.bigru:
                    print('[FULL TRAIN] BiGRU Model')
                    train_bigru_model(full_train_seq_sentences, full_train_seq_labels, dev_seq_sentences, dev_seq_labels, args.version)

                if args.bilstm:
                    print('[FULL TRAIN] BiLSTM Model')
                    train_bilstm_model(full_train_seq_sentences, full_train_seq_labels, dev_seq_sentences, dev_seq_labels, args.version)

        if args.test:
            test_data_type = ['test.' + args.data]

            if args.ffnn:
                print('[TEST] Loading data for dataset {}'.format(args.data))
                test_sentences, test_labels = Preprocessor.load_indosum_data_by_sentence(sentence_embedding, type=test_data_type)
                print('[TEST] Finished loading data')

                print('[TEST] FFNN Model')
                load_and_test_ffnn_model(test_sentences, test_labels, args.version)

            if args.gru or args.lstm or args.bigru or args.bilstm:
                print('[TEST] Loading sequence data for dataset {}'.format(args.data))
                test_seq_sentences, test_seq_labels = Preprocessor.load_indosum_data(sentence_embedding, type=test_data_type)
                print('[TEST] Finished loading sequence data')

                if args.gru:
                    print('[TEST] GRU Model')
                    load_and_test_gru_model(test_seq_sentences, test_seq_labels, args.version)

                if args.lstm:
                    print('[TEST] LSTM Model')
                    load_and_test_lstm_model(test_seq_sentences, test_seq_labels, args.version)

                if args.bigru:
                    print('[TEST] BiGRU Model')
                    load_and_test_bigru_model(test_seq_sentences, test_seq_labels, args.version)

                if args.bilstm:
                    print('[TEST] BiLSTM Model')
                    load_and_test_bilstm_model(test_seq_sentences, test_seq_labels, args.version)
