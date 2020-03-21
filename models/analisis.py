#!/usr/bin/env python3

import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import cleaning
import glove
import data
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import Counter
from collections import OrderedDict
from typing import Any
from typing import Dict
from typing import Set
from typing import List
from typing import Tuple

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


def print_samples(dataset: pd.DataFrame) -> None:
    """ Print entries """
    print(dataset.head(3))


def plot_labels_distribution(dataset: pd.DataFrame) -> None:
    """ Plot labels distribution """
    counts = train.target.value_counts()
    sns.barplot(counts.index, counts.values)
    plt.gca().set_ylabel("samples")
    plt.show()


def plot_number_of_characters(dataset: pd.DataFrame) -> None:
    """ Plot characters distribution """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    tweet_len = dataset[dataset['target'] == 1]['text'].str.len()
    sns.distplot(tweet_len, ax=ax1, color='red')
    ax1.set_title('disaster tweets')

    tweet_len = dataset[dataset['target'] == 0]['text'].str.len()
    sns.distplot(tweet_len, ax=ax2, color='green')
    ax2.set_title('not disaster tweets')
    fig.suptitle('Length of tweets')
    plt.show()


def plot_number_of_words(dataset: pd.DataFrame) -> None:
    """ Plot number of words in tweets """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    tweet_len = dataset[dataset['target'] == 1]['text'].str.split().map(lambda x: len(x))
    sns.distplot(tweet_len, ax=ax1, color='red')
    ax1.set_title('disaster tweets')

    tweet_len = dataset[dataset['target'] == 0]['text'].str.split().map(lambda x: len(x))
    sns.distplot(tweet_len, ax=ax2, color='green')
    ax2.set_title('not disaster tweets')
    fig.suptitle('Number of words in tweet')
    plt.show()


def plot_average_word_length(dataset: pd.DataFrame) -> None:
    """ Plot average word length """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    word_len = dataset[dataset['target'] == 1]['text'].str.split().map(lambda x: [len(i) for i in x])
    sns.distplot(word_len.map(lambda x: np.mean(x)), ax=ax1, color='red')
    ax1.set_title('disaster tweets')

    word_len = dataset[dataset['target'] == 0]['text'].str.split().map(lambda x: [len(i) for i in x])
    sns.distplot(word_len.map(lambda x: np.mean(x)), ax=ax2, color='green')
    ax2.set_title('not disaster tweets')
    fig.suptitle('Averge word length')
    plt.show()


def _make_word_count_dict(corpus: List[str], set_: Set[str]) -> Dict[str, int]:
    """ Count amount of words from set_ in corpus """
    dst = defaultdict(int)
    for word in corpus:
        if word in set_:
            dst[word] += 1
    return dst
    

def _make_word_count_df(src: Dict[str, int]) -> pd.DataFrame:
    """ Convet word count dict to DataFrame """
    dict_ = {"word": [], "count": []}
    for key, value in src.items():
        dict_["word"].append(key)
        dict_["count"].append(value)
    return pd.DataFrame(dict_)


def _plot_stop_words_distribution(ax: Any, color: str, dataset: pd.DataFrame, target: int) -> None:
    """ Plot common stop words """
    stop_words = set(stopwords.words('english'))
    corpus = data.create_corpus(dataset, target)
    count_dict = _make_word_count_dict(corpus, stop_words) 
    count_dict_top = dict(sorted(count_dict.items(), key=lambda x:x[1], reverse=True)[:10])
    count_df = _make_word_count_df(count_dict_top)
    sns.barplot(x="word", y="count", data=count_df, ax=ax, color=color)
    ax.set_title("target = " + str(target))


def plot_stop_words_distribution(dataset: pd.DataFrame) -> None:
    """ Plot common stop words for both classes of dataset """
    sns.set()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    _plot_stop_words_distribution(ax1, "green", dataset, 0)
    _plot_stop_words_distribution(ax2, "red", dataset, 1)
    fig.suptitle("Top stop words")
    plt.show()
    

def _plot_punctuation_distribution(ax: Any, color: str, dataset: pd.DataFrame, target: int) -> None:
    """ Plot frequency of punctuation characters impl """
    corpus = data.create_corpus(dataset, target)
    count = _make_word_count_dict(corpus, string.punctuation)
    count_df = _make_word_count_df(count)
    sns.barplot(x="word", y="count", data=count_df, ax=ax, color=color)
    ax.set_title("target = " + str(target))

    
def plot_punctuation_distribution(dataset: pd.DataFrame) -> None:
    """ Plot frequency of punctuation characters """
    sns.set()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    _plot_punctuation_distribution(ax1, "green", dataset, 0)
    _plot_punctuation_distribution(ax2, "red", dataset, 1)
    fig.suptitle("Punctuation distribution")
    plt.show()


def plot_most_common_words(dataset: pd.DataFrame) -> None:
    """ Plot common words """
    stop_words = set(stopwords.words('english'))
    corpus = data.create_corpus_all(dataset)
    corpus = [w for w in corpus if w not in stop_words]
    counter = Counter(corpus)
    most_common_dict = OrderedDict(sorted(counter.most_common(20), key=lambda x: x[1], reverse=True))
    most_common_df = _make_word_count_df(most_common_dict)
    sns.barplot(x="count", y="word", data=most_common_df)
    plt.show()


def get_top_tweets_bigrams(dataset: pd.DataFrame, n: int=None) -> List[Tuple[str, int]]:
    """ Get top n bigrams with frequencies """
    corpus = data.create_corpus_all(dataset)
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(w, sum_words[0, i]) for w, i in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


def plot_bigrams_distribution(dataset: pd.DataFrame) -> None:
    """ Plot distribution of bigrams """
    plt.figure(figsize=(10, 5))
    top_tweet_bigrams = get_top_tweets_bigrams(dataset)[:20]
    x, y = map(list, zip(*top_tweet_bigrams))
    sns.barplot(x=y, y=x)
    plt.show()


def get_words_not_covered_by_embeddings(corpus: Set[str]) -> List[str]:
    """ Return words that are not presented in embeddings table """
    embeddings_dict = glove.glove_6b_dict()
    return [w for w in corpus if w not in embeddings_dict]


def get_corpus_dict(dataset: pd.DataFrame) -> Dict[str, int]:
    """ Get word to frequency dict """
    corpus = data.create_glove_corpus(dataset)
    corpus = list(itertools.chain(*corpus))
    corpus_dict = defaultdict(int)
    for word in corpus:
        corpus_dict[word] += 1
    return corpus_dict


def print_embeddings_statistics(dataset: pd.DataFrame) -> None:
    """ Print statistics about words not covered by embeddings """
    corpus_dict = get_corpus_dict(dataset)
    not_covered = get_words_not_covered_by_embeddings(corpus_dict.keys())
    not_covered_dict = {w: corpus_dict[w] for w in not_covered}
    not_covered_sorted = sorted(not_covered_dict.items(), key=lambda x: x[1], reverse=True)
    
    for word, count in not_covered_sorted[0:50]:
        print("word: " + word + "; count: " + str(count))
        
    print("Not covered: " + str(len(not_covered)))
    print("Corpus: " + str(len(corpus_dict)))
    print("Ratio: " + str(len(not_covered) / len(corpus_dict)))

                                        
if __name__ == "__main__":
    train, test = data.load(train="train.cleaned.csv")
    print_embeddings_statistics(train)

    
