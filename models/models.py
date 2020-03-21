#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import SpatialDropout1D
from keras.initializers import Constant
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from typing import Any
import bert_tools
import data


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def make_lstm_64(embeddings_matrix: np.ndarray, tweet_max_len: int) -> Any:
    """ Make simple lstm model """
    embedding_len = embeddings_matrix.shape[1]
    embedding_args = {}
    embedding_args["embeddings_initializer"] = Constant(embeddings_matrix)
    embedding_args["input_length"] = tweet_max_len
    embedding_args["trainable"] = False
    embedding = Embedding(embeddings_matrix.shape[0], embedding_len, **embedding_args)
    model = Sequential()
    model.add(embedding)
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(128, dropout=0.4, recurrent_dropout=0.4))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(lr=1e-5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', f1_m])
    model.summary()
    return model


def make_bert_model(n_fine_tune_layers: int=1) -> Any:
    """ Make tweets classification model based on BERT emncoding """
    input_ids = tf.keras.Input(shape=(data.TWEET_MAX_LEN,), name="input_ids")
    input_mask = tf.keras.Input(shape=(data.TWEET_MAX_LEN,), name="input_mask")
    input_segment = tf.keras.Input(shape=(data.TWEET_MAX_LEN,), name="input_segment")
    bert_inputs = [input_ids, input_mask, input_segment]
    bert_output = bert_tools.BertLayer(n_fine_tune_layers=n_fine_tune_layers, pooling="first")(bert_inputs)
    dropout = tf.keras.layers.Dropout(rate=0.1)(bert_output)
    dense = tf.keras.layers.Dense(256, activation="relu")(dropout)
    pred = tf.keras.layers.Dense(1, activation="sigmoid")(dense)

    optimizer = tf.keras.optimizers.Adam(lr=5e-6)
    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', f1_m])
    model.summary()
    return model


def make_bert_v2_model(n_fine_tune_layers: int=1, pooling: str="first") -> Any:
    """ Make tweets classification model based on BERT emncoding """
    l2 = tf.keras.regularizers.l2(0.05)
    input_ids = tf.keras.Input(shape=(data.TWEET_MAX_LEN,), name="input_ids")
    input_mask = tf.keras.Input(shape=(data.TWEET_MAX_LEN,), name="input_mask")
    input_segment = tf.keras.Input(shape=(data.TWEET_MAX_LEN,), name="input_segment")
    bert_inputs = [input_ids, input_mask, input_segment]
    bert_output = bert_tools.BertLayer(n_fine_tune_layers=n_fine_tune_layers, pooling=pooling)(bert_inputs)
    dropout0 = tf.keras.layers.Dropout(rate=0.25)(bert_output)
    dense0 = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=l2)(dropout0)
    dropout1 = tf.keras.layers.Dropout(rate=0.25)(dense0)
    dense1 = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=l2)(dropout1)
    dropout2 = tf.keras.layers.Dropout(rate=0.25)(dense1)
    pred = tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=l2)(dropout2)

    optimizer = tf.keras.optimizers.Adam(lr=1e-5)
    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', f1_m])
    model.summary()
    return model


if __name__ == "__main__":
    model = make_bert_model()

