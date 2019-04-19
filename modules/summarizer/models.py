import numpy as np
import pickle

from abc import ABC
from keras.layers import Input, Dense, TimeDistributed
from keras.layers import LeakyReLU, Dropout, Bidirectional
from keras.layers import GRU, LSTM
from keras.models import Model
from keras.models import load_model

from modules.constants import Constants
from modules.sentence_embedding import SentenceEmbedding
from modules.summarizer.metrics import precision, recall, f1
from modules.summarizer.preprocessor import Preprocessor
from modules.utils.text_util import split_to_sentences


class BaseModel(ABC):
    def __init__(self):
        self._model = None

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, log_path: str):
        history = self._model.fit(
            x_train,
            y_train,
            epochs=100,
            validation_data=(x_val, y_val)
        )
        logfile = open(Constants.MODEL_PATH + log_path, 'wb+')
        pickle.dump(history.history, logfile)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._model.predict(x)

    def summarize(self, text: str, sentence_embedding: SentenceEmbedding, n_sentence: int = 0) -> str:
        sentences = split_to_sentences(text)
        last_sentence_idx = len(sentences) - 1

        sentences_vector = Preprocessor.preprocess_text(sentences, sentence_embedding)
        model_predictions = self.predict(sentences_vector)

        predictions = model_predictions[0]

        index_confidence_list = []
        for idx, prediction in enumerate(predictions):
            index_confidence_list.append((idx, prediction[1]))

        selected_indices = []

        if n_sentence > 0:
            n = 0
            index_confidence_list.sort(key=lambda x: x[1], reverse=True)
            print(index_confidence_list)
            while n < n_sentence:
                if index_confidence_list[n][0] <= last_sentence_idx:
                    selected_indices.append(index_confidence_list[n][0])
                    n += 1
                else:
                    print('WARNING')
                    print(index_confidence_list[n][0])
                    print(index_confidence_list[n][1])
                    print()
                    n += 1
                    n_sentence += 1
            selected_indices.sort()
        else:  # Use threshold
            for idx, confidence in index_confidence_list:
                if confidence > Constants.MODEL_THRESHOLD and idx < last_sentence_idx:
                    selected_indices.append(idx)

        summary = ''

        for selected_index in selected_indices:
            summary += sentences[selected_index] + ' '

        summary = summary.strip()
        return summary

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> (float, float, float):
        loss, accuracy_score, precision_score, recall_score, f1_score = self._model.evaluate(x_test, y_test)
        return loss, accuracy_score, precision_score, recall_score, f1_score

    def save(self, model_name: str):
        self._model.save(Constants.MODEL_PATH + '{}.h5'.format(model_name))
        print('Model saved to {}'.format(Constants.MODEL_PATH + '{}.h5'.format(model_name)))

    def load(self, model_name: str):
        self._model = load_model(Constants.MODEL_PATH + '{}.h5'.format(model_name), custom_objects={'precision': precision, 'recall': recall, 'f1': f1})


class FFNNModel(BaseModel):
    def __init__(self):
        super().__init__()
        inputs = Input(shape=(Constants.WORD_EMBEDDING_DIMENSION,))

        hidden_1 = Dense(256)(inputs)
        actvfn_1 = LeakyReLU(alpha=0.1)(hidden_1)
        drpout_1 = Dropout(0.5)(actvfn_1)

        hidden_2 = Dense(128)(drpout_1)
        actvfn_2 = LeakyReLU(alpha=0.1)(hidden_2)
        drpout_2 = Dropout(0.5)(actvfn_2)

        predictions = Dense(2, activation='softmax')(drpout_2)

        self._model = Model(inputs=inputs, outputs=predictions)
        self._model.summary()

        self._model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', precision, recall, f1]
        )


class GRUModel(BaseModel):
    def __init__(self):
        super().__init__()
        inputs = Input(shape=(Constants.MAXIMUM_SENTENCE_LENGTH, Constants.WORD_EMBEDDING_DIMENSION,))

        input_dropout = Dropout(0.5)(inputs)

        gru_1 = GRU(
            64,
            dropout=0.5,
            recurrent_dropout=0.5,
            return_sequences=True
        )(input_dropout)                

        gru_2 = GRU(
            64,
            dropout=0.5,
            recurrent_dropout=0.5,
            return_sequences=True
        )(gru_1)

        predictions = TimeDistributed(Dense(
            2,
            activation='softmax'
        ))(gru_2)

        self._model = Model(inputs=inputs, outputs=predictions)
        self._model.summary()

        self._model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', precision, recall, f1]
        )


class LSTMModel(BaseModel):
    def __init__(self):
        super().__init__()
        inputs = Input(shape=(Constants.MAXIMUM_SENTENCE_LENGTH, Constants.WORD_EMBEDDING_DIMENSION,))

        input_dropout = Dropout(0.5)(inputs)

        lstm_1 = LSTM(
            64,
            dropout=0.5,
            recurrent_dropout=0.5,
            return_sequences=True
        )(input_dropout)

        lstm_2 = LSTM(
            64,
            dropout=0.5,
            recurrent_dropout=0.5,
            return_sequences=True
        )(lstm_1)

        predictions = TimeDistributed(Dense(
            2,
            activation='softmax'
        ))(lstm_2)

        self._model = Model(inputs=inputs, outputs=predictions)
        self._model.summary()

        self._model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', precision, recall, f1]
        )


class BiGRUModel(BaseModel):
    def __init__(self):
        super().__init__()
        inputs = Input(shape=(Constants.MAXIMUM_SENTENCE_LENGTH, Constants.WORD_EMBEDDING_DIMENSION,))

        input_dropout = Dropout(0.5)(inputs)

        bigru_1 = Bidirectional(GRU(
            64,
            dropout=0.5,
            recurrent_dropout=0.5,
            return_sequences=True
        ))(input_dropout)

        bigru_2 = Bidirectional(GRU(
            64,
            dropout=0.5,
            recurrent_dropout=0.5,
            return_sequences=True
        ))(bigru_1)

        predictions = TimeDistributed(Dense(
            2,
            activation='softmax'
        ))(bigru_2)

        self._model = Model(inputs=inputs, outputs=predictions)
        self._model.summary()

        self._model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', precision, recall, f1]
        )


class BiLSTMModel(BaseModel):
    def __init__(self):
        super().__init__()
        inputs = Input(shape=(Constants.MAXIMUM_SENTENCE_LENGTH, Constants.WORD_EMBEDDING_DIMENSION,))

        input_dropout = Dropout(0.5)(inputs)

        bilstm_1 = Bidirectional(LSTM(
            64,
            dropout=0.5,
            recurrent_dropout=0.5,
            return_sequences=True
        ))(input_dropout)

        bilstm_2 = Bidirectional(LSTM(
            64,
            dropout=0.5,
            recurrent_dropout=0.5,
            return_sequences=True
        ))(bilstm_1)

        predictions = TimeDistributed(Dense(
            2,
            activation='softmax'
        ))(bilstm_2)

        self._model = Model(inputs=inputs, outputs=predictions)
        self._model.summary()

        self._model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', precision, recall, f1]
        )
