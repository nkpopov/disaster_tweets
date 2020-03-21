#!/usr/bin/env python3

import tensorflow as tf
from typing import Tuple
import numpy as np
import pandas as pd
from tensorflow import keras
import cleaning
import models
import data
import os


def get_trainingset() -> Tuple:
    """ Get training and validation dataset """
    train, test = data.load(train="train.cleaned.csv")
    seq, matrix = data.create_sequences(train, data.TWEET_MAX_LEN)
    x_train, x_val, y_train, y_val = data.train_val_split(seq, train, test_size=0.05)
    return matrix, x_train, x_val, y_train, y_val


def get_bert_trainingset_aug(sess: tf.Session) -> Tuple:
    """ Get training and validation dataset """
    train_cln, _ = data.load(train="train.cleaned.csv")
    train_raw, _ = data.load(train="train.csv")
    train = pd.concat([train_cln, train_raw]).sort_index().reset_index(drop=True)
    tokenizer = data.create_bert_tokenizer(sess)
    bert_input = data.get_bert_input(tokenizer, train)
    split = data.train_val_split_bert(bert_input, train['target'].values, test_size=0.05)
    return split


def get_bert_trainingset(sess: tf.Session) -> Tuple:
    """ Get training and validation dataset """
    train, _ = data.load(train="train.csv")
    tokenizer = data.create_bert_tokenizer(sess)
    bert_input = data.get_bert_input(tokenizer, train)
    split = data.train_val_split_bert(bert_input, train['target'].values, test_size=0.05)
    return split


def initialize_vars(sess: tf.Session()) -> None:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())


def get_ckpt_callback(ckpt_path: str) -> tf.keras.callbacks.ModelCheckpoint:
    """ Get checkpoint callback """
    ckpt_dir = os.path.dirname(ckpt_path)
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
        monitor='val_loss', save_best_only=False, mode='auto', period=1,
        save_weights_only=False, verbose=1)
    return ckpt_callback

    
if __name__ == "__main__":

    if False:
        matrix, x_tr, x_val, y_tr, y_val = get_trainingset()
        model = models.make_lstm_64(matrix, data.TWEET_MAX_LEN)
        history = model.fit(x_tr, y_tr, batch_size=4, epochs=100, validation_data=(x_val, y_val))
        model.save("../logs/lstm_128_with_better_embeddings_coverage_dropout_0.4.h5")

    if False:
        ckpt_path = "../checkpoints/bert_256_2_{epoch:03d}_{val_f1_m:.4f}.h5"
        ckpt_callback = get_ckpt_callback(ckpt_path)
        sess = tf.Session()
        keras.backend.set_session(sess)
        x_tr, x_val, y_tr, y_val = get_bert_trainingset(sess)
        model = models.make_bert_model(n_fine_tune_layers=2)
        initialize_vars(sess)
        history = model.fit(x_tr, y_tr, batch_size=8, epochs=50,
            validation_data=(x_val, y_val), callbacks=[ckpt_callback])
        model.save("../logs/bert_256_2.h5")

    if True:
        ckpt_path = "../checkpoints/bert_v2_2layers_mean_{epoch:03d}_{val_f1_m:.4f}.h5"
        ckpt_callback = get_ckpt_callback(ckpt_path)
        sess = tf.Session()
        keras.backend.set_session(sess)
        x_tr, x_val, y_tr, y_val = get_bert_trainingset(sess)
        model = models.make_bert_v2_model(n_fine_tune_layers=2, pooling="mean")
        initialize_vars(sess)
        history = model.fit(x_tr, y_tr, batch_size=8, epochs=100,
            validation_data=(x_val, y_val), callbacks=[ckpt_callback])
        model.save("../logs/bert_v2_2layers_mean.h5")
    

