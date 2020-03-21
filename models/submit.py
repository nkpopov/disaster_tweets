#!/usr/bin/env python3

import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import cleaning
import models
import train
import data
from typing import Any
from typing import List


SAMPLE_SUBMISSION="../data/sample_submission.csv"


def predictions_to_dataframe(predictions: np.ndarray) -> pd.DataFrame:
    """ Convert predictions to pd.DataFrame format """
    y = np.round(predictions).astype(int).reshape(-1)
    sample_sub = pd.read_csv(SAMPLE_SUBMISSION)
    sample_ids = sample_sub['id'].values.tolist()
    return pd.DataFrame({"id": sample_ids, "target": y})


def get_testset() -> pd.DataFrame:
    """ Return preprocessed testset """

    test, _ = data.create_sequences(test, data.TWEET_MAX_LEN)
    return test


def submit(model_path: str, dst: str):
    """ Evaluate model and submit results """
    model = keras.models.load_model(model_path)
    testset = get_testset()
    predictions = model.predict(testset)
    df = predictions_to_dataframe(predictions)
    df.to_csv(dst, index=False)
    print("Success; model: " + model_path + "; dst: " + dst)
    print(df.head())


def get_bert_testset(sess: tf.Session) -> Any:
    """ Get preprocessed testset """
    _, test = data.load(test="test.csv")
    tokenizer = data.create_bert_tokenizer(sess)
    bert_input = data.get_bert_input(tokenizer, test)
    return bert_input


def submit_bert(model_path: str, dst: str) -> None:
    """ Evaluate model and submit results """
    sess = tf.Session()
    keras.backend.set_session(sess)
    testset = get_bert_testset(sess)
    model = models.make_bert_v2_model(n_fine_tune_layers=2)
    train.initialize_vars(sess)
    model.load_weights(model_path)
    predictions = model.predict(testset)
    df = predictions_to_dataframe(predictions)
    df.to_csv(dst, index=False)
    print("Success; model: " + model_path + "; dst: " + dst)
    print(df.head())

    
def submit_bert_combined(model_paths: List[str], dst: str) -> None:
    """ Evaluate model and submit results """
    sess = tf.Session()
    keras.backend.set_session(sess)
    testset = get_bert_testset(sess)
    model = models.make_bert_v2_model(n_fine_tune_layers=2, pooling="mean")
    train.initialize_vars(sess)

    predictions_list = []
    for w_path in model_paths:
        model.load_weights(w_path)
        predictions_list.append(model.predict(testset))

    predictions_list = np.array(predictions_list)
    print(predictions_list.shape)
    predictions = np.average(np.array(predictions_list), axis=0)
    print(predictions.shape)
    
    df = predictions_to_dataframe(predictions)
    df.to_csv(dst, index=False)
    print("Success; model: " + str(model_paths) + "; dst: " + dst)
    print(df.head())

    
if __name__ == "__main__":
    if False:
        submit(
            "../logs/lstm_64_without_cleaning.h5",
            "../logs/lstm_64_without_cleaning.csv"
        )
    if False:
        submit(
            "../logs/lstm_64_with_cleaning.h5",
            "../logs/lstm_64_with_cleaning.csv"
        )
    if False:
        submit(
            "../logs/lstm_64_with_cleaning_dropout_0.1.h5",
            "../logs/lstm_64_with_cleaning_dropout_0.1.csv"
        )
    if False:
        submit(
            "../logs/lstm_128_with_better_embeddings_coverage.h5",
            "../logs/lstm_128_with_better_embeddings_coverage.csv"
        )
    if False:
        submit_bert(
            "../logs/bert_256_with_cleaned_data.h5",
            "../logs/bert_256_with_cleaned_data.csv"
        )
    if False:
        submit_bert(
            "../logs/bert_256_2_06_0.87.h5",
            "../logs/bert_256_2_06_0.87.csv"
        )
    if False:
        submit_bert(
            "../logs/bert_v2_64_0.85.h5",
            "../logs/bert_v2_64_0.85.csv"
        )
    if False:
        submit_bert(
            "../logs/bert_v2_2layers_050_0.8639.h5",
            "../logs/bert_v2_2layers_050_0.8639.csv"
        )
    if False:
        submit_bert(
            "../logs/bert_v2_2layers_011_0.8407.h5",
            "../logs/bert_v2_2layers_011_0.8407.csv"
        )
    if False:
        submit_bert_combined(
            ["../checkpoints/bert_v2_2layers_050_0.8639.h5",
             "../checkpoints/bert_v2_2layers_060_0.8546.h5"],
            "../logs/bert_v2_2layers_combined.csv"
        )
    if False:
        submit_bert_combined(
            ["../checkpoints/bert_v2_2layers_dirty_005_0.8523.h5",
             "../checkpoints/bert_v2_2layers_dirty_003_0.8414.h5",
             "../checkpoints/bert_v2_2layers_dirty_006_0.8419.h5",
             "../checkpoints/bert_v2_2layers_dirty_007_0.8427.h5",
             "../checkpoints/bert_v2_2layers_dirty_012_0.8443.h5",
             "../checkpoints/bert_v2_2layers_dirty_013_0.8402.h5",
             "../checkpoints/bert_v2_2layers_dirty_015_0.8446.h5",
             "../checkpoints/bert_v2_2layers_dirty_017_0.8448.h5",
             "../checkpoints/bert_v2_2layers_dirty_019_0.8485.h5"],
            "../logs/bert_v2_2layers_dirty_combined.csv"
        )
    if True:
        submit_bert_combined(
            ["../checkpoints/bert_v2_2layers_mean_007_0.8531.h5",
             "../checkpoints/bert_v2_2layers_mean_008_0.8508.h5",
             "../checkpoints/bert_v2_2layers_mean_012_0.8504.h5",
             "../checkpoints/bert_v2_2layers_mean_009_0.8434.h5",
             "../checkpoints/bert_v2_2layers_mean_010_0.8404.h5",
             "../checkpoints/bert_v2_2layers_mean_016_0.8466.h5",
             "../checkpoints/bert_v2_2layers_mean_017_0.8447.h5",
            ],
            "../logs/bert_v2_2layers_mean_combined.csv"
        )
