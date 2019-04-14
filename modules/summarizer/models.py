import numpy as np
import pickle

from abc import ABC
from keras.layers import Input, Dense, TimeDistributed
from keras.layers import LeakyReLU, Dropout, Bidirectional
from keras.layers import GRU, LSTM
from keras.models import Model
from keras.models import load_model

from modules.constants import Constants
from modules.summarizer.metrics import precision, recall, f1


class BaseModel(ABC):
    def __init__(self):
        self._model = None

    def train(self, x_train: np.ndarray, y_train: np.ndarray, log_path: str):
        history = self._model.fit(
            x_train,
            y_train,
            epochs=50
        )
        logfile = open(Constants.MODEL_PATH + log_path, 'wb+')
        pickle.dump(history.history, logfile)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._model.predict(x)

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
        drpout_1 = Dropout(0.1)(actvfn_1)

        hidden_2 = Dense(128)(drpout_1)
        actvfn_2 = LeakyReLU(alpha=0.1)(hidden_2)
        drpout_2 = Dropout(0.1)(actvfn_2)

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
        inputs = Input(shape=(35, Constants.WORD_EMBEDDING_DIMENSION,))

        gru_1 = GRU(
            64,
            dropout=0.1,
            recurrent_dropout=0.1,
            return_sequences=True
        )(inputs)

        gru_2 = GRU(
            64,
            dropout=0.1,
            recurrent_dropout=0.1,
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
        inputs = Input(shape=(35, Constants.WORD_EMBEDDING_DIMENSION,))

        lstm_1 = LSTM(
            64,
            dropout=0.1,
            recurrent_dropout=0.1,
            return_sequences=True
        )(inputs)

        lstm_2 = LSTM(
            64,
            dropout=0.1,
            recurrent_dropout=0.1,
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
        inputs = Input(shape=(35, Constants.WORD_EMBEDDING_DIMENSION,))

        gru_1 = Bidirectional(GRU(
            64,
            dropout=0.1,
            recurrent_dropout=0.1,
            return_sequences=True
        ))(inputs)

        gru_2 = Bidirectional(GRU(
            64,
            dropout=0.1,
            recurrent_dropout=0.1,
            return_sequences=True
        ))(gru_1)

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


class BiLSTMModel(BaseModel):
    def __init__(self):
        super().__init__()
        inputs = Input(shape=(35, Constants.WORD_EMBEDDING_DIMENSION,))

        lstm_1 = Bidirectional(LSTM(
            64,
            dropout=0.1,
            recurrent_dropout=0.1,
            return_sequences=True
        ))(inputs)

        lstm_2 = Bidirectional(LSTM(
            64,
            dropout=0.1,
            recurrent_dropout=0.1,
            return_sequences=True
        ))(lstm_1)

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
