#!/usr/bin/env python3

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from typing import Tuple
from typing import List
from typing import Any
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from bert.tokenization import FullTokenizer
from typing import Dict
import bert_tools
import glove


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# Maximum amount of words in a tweet
TWEET_MAX_LEN = 50



def load(root: str="../data/", train: str="train.csv", test: str="test.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Load dataset from csv files """
    train = pd.read_csv(root + train)
    test  = pd.read_csv(root + test)

    print("Train: %d rows, %d cols" % (train.shape[0], train.shape[1]))
    print("Test:  %d rows, %d cols" % (test.shape[0], test.shape[1]))
    return train, test


def create_corpus(dataset: pd.DataFrame, target: int) -> List[str]:
    """ Create of words """
    corpus = []
    tweets = dataset[dataset['target'] == target]['text'].str.split()
    for tweet in tweets:
        corpus.extend([w for w in tweet])
    return corpus


def create_corpus_all(dataset: pd.DataFrame) -> List[str]:
    """ Create of words """
    corpus = []
    tweets = dataset['text'].str.split()
    for tweet in tweets:
        corpus.extend([w for w in tweet])
    return corpus


def create_glove_corpus(dataset: pd.DataFrame) -> List[str]:
    """ Create corpus for Glove embeddings """
    corpus = []
    stop_words = set(stopwords.words('english'))

    print("create glove corpus")
    for tweet in tqdm(dataset['text']):
        words = [w.lower() for w in word_tokenize(tweet)]
        words = [w for w in words if w.isalpha()]
        words = [w for w in words if w not in stop_words]
        corpus.append(words)
    return corpus


def create_glove_6b_embeddings_matrix(word_index: Dict[str, int]) -> np.ndarray:
    """ Create [num_words, embedding_len] matrix for glove 6b embeddings """

    # We meed to add one, since word indices start with 1
    num_words = len(word_index) + 1
    embeddings_dict = glove.glove_6b_dict()
    embedding_len = len(next(iter(embeddings_dict.values())))
    embeddings_matrix = np.zeros((num_words, embedding_len))

    print("create embedding matrix")
    for word, i in tqdm(word_index.items()):
        embeddings_matrix[i] = embeddings_dict.get(word, np.zeros((1, embedding_len)))
    return embeddings_matrix
    

def create_sequences(dataset: pd.DataFrame, maxlen=50) -> Tuple[List[List[int]], np.ndarray]:
    """ Convert dataset to GloVe 6B embeddings matrix """

    # dims: [num_tweets, tweet_len]
    corpus = create_glove_corpus(dataset)

    # dims: [num_tweets, maxlen]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    sequences = tokenizer.texts_to_sequences(corpus)
    sequences = pad_sequences(sequences, maxlen=maxlen, truncating='post', padding='post')

    # dims: [num_words, embedding_len]
    matrix = create_glove_6b_embeddings_matrix(tokenizer.word_index)
    return sequences, matrix


def train_val_split(seq: np.array, dataset: pd.DataFrame, test_size: float) -> Tuple:
    """ Split dataset on train & test samples & labels """
    labels = dataset['target'].values
    x_train, x_test, y_train, y_test = train_test_split(seq, labels, test_size=test_size)

    print("Shape of train: ", x_train.shape)
    print("Shape of validation: ", x_test.shape)
    return (x_train, x_test, y_train, y_test)


def create_bert_tokenizer(sess: tf.Session):
    """ Create tokenizer for bert encoding """
    bert_module = hub.Module(bert_tools.BERT_PATH)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"]
        ]
    )

    print(vocab_file)
    print(do_lower_case)
    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def get_bert_tokens(tokenizer: FullTokenizer, text: str) -> Any:
    """ Create BERT tokens for input text """
    def _create_tokens(tokenizer: FullTokenizer, text: str) -> List[str]:
        return ["[CLS]"] + tokenizer.tokenize(text)[0:TWEET_MAX_LEN-2] + ["[SEP]"]
    def _create_input_ids(tokens: List[str]) -> List[int]:
        return tokenizer.convert_tokens_to_ids(tokens)
    def _create_input_mask(tokens: List[str]) -> List[int]:
        return [1 for _ in range(len(tokens))]
    def _create_segments_ids(tokens: List[str]) -> List[int]:
        return [0 for _ in range(len(tokens))]
    def _zero_pad(src: List[Any], size: int) -> List[Any]:
        return src + [0] * (size - len(src))
    
    tokens = _create_tokens(tokenizer, text)
    input_ids = _zero_pad(_create_input_ids(tokens), TWEET_MAX_LEN)
    input_mask = _zero_pad(_create_input_mask(tokens), TWEET_MAX_LEN)
    segments_ids = _zero_pad(_create_segments_ids(tokens), TWEET_MAX_LEN)
    return input_ids, input_mask, segments_ids


def get_bert_input(tokenizer: FullTokenizer, dataset: pd.DataFrame) -> Tuple:
    """ Create input for BERT encoding """
    input_ids_vec, input_mask_vec, segments_ids_vec = [], [], []
    for text in dataset['text']:
        input_ids, input_mask, segments_ids = get_bert_tokens(tokenizer, text)
        input_ids_vec.append(input_ids)
        input_mask_vec.append(input_mask)
        segments_ids_vec.append(segments_ids)
    return (
        np.array(input_ids_vec),
        np.array(input_mask_vec),
        np.array(segments_ids_vec)
    )

def train_val_split_bert(dataset: Tuple, labels: List, test_size: float) -> Tuple:
    """ Split dataset on train & test samples & labels """
    dataset = np.array(dataset).transpose([1, 0, 2])
    x_index = int(len(dataset) * (1 - test_size))
    x_train, y_train = dataset[0:x_index], labels[0:x_index]
    x_test, y_test = dataset[x_index:], labels[x_index:]
    x_train = x_train.transpose([1, 0, 2])
    x_test = x_test.transpose([1, 0, 2])

    print("Shape of train: ", x_train.shape, y_train.shape)
    print("Shape of validation: ", x_test.shape, y_test.shape)

    x_train = (x_train[0], x_train[1], x_train[2])
    x_test = (x_test[0], x_test[1], x_test[2])
    return (x_train, x_test, y_train, y_test)
    

if __name__ == "__main__":
    sess = tf.Session()
    dataset = pd.DataFrame({"text": ["Cat is lying of the couchh"]})
    tokenizer = create_bert_tokenizer(sess)
    bert_input = get_bert_input(tokenizer, dataset)
    print(bert_input)
    
