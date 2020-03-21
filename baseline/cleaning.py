#!/usr/bin/env python3

import string
import pandas as pd
import re
import data
from spellchecker import SpellChecker
from typing import Callable
from tqdm import tqdm


EMOJIS = [
    u"\U0001F600-\U0001F64F", # emoticons
    u"\U0001F300-\U0001F5FF", # symbols
    u"\U0001F680-\U0001F6FF", # map symbols
    u"\U0001F1E0-\U0001F1FF", # flags
    u"\U00002702-\U000027B0",
    u"\U000024C2-\U0001F251",
]

UNKNOWN_WORDS = {
    "typhoondevastated": "typhoon devastated",
    "bestnaijamade": "musician",
    "gbbo": "great british bake off",
    "dista": "parties",
    "reddits": "websites",
    "funtenna": "software",
    "subreddits": "websites",
    "nowplaying": "now playing",
    "sensorsenso": "rock band",
    "arianagrande": "woman singer",
    "sismo": "place",
    "spos": "supposedly",
    "directioners": "fandoms",
    "trfc": "soccer team",
    "worldnews": "world news",
    "justinbieber": "man singer",
    "beyhive": "shop",
    "mediterran": "southern europe",
    "hwo": "who",
    "irandeal": "iran deal",
    "trapmusic": "trap music",
    "linkury": "software",
    "icemoon": "ice moon",
    "djicemoon": "dj ice moon",
    "animalrescue": "animal rescue",
    "prophetmuhammad": "prophet muhammad",
    "tubestrike": "tube strike",
    "mikeparractor": "man actor",
    "chicagoarea": "chicago area",
    "igers": "users of website",
    "standuser": "stand user",
    "geddit": "web site",
    "mtvhottest": "hottest content of television channel",
    "meatloving": "meat loving",
    "abcnews": "news channel",
    "viralspell": "website with historical records and family trees",
    "socialnews": "social news",
    "summerfate": "photo of a face during summer",
    "stoday": "today",
    "kerricktrial": "legal trial",
    "collisionno": "collistion no",
    "usagov": "government",
    "pantherattack": "panthera attack",
    "nasahurricane": "hurricane",
    "gtgtgt": "funny images",
    "darude": "musician",
    "youngheroesid": "young heroes",
    "explosionproof": "explosion resistant",
    "strategicpatience": "strategic patience",
    "worstsummerjob": "worst summer job"
}


def _copy_and_apply(dataset: pd.DataFrame, func: Callable) -> pd.DataFrame:
    """ Copy dataset and apply function to the text of a copied version """
    result = dataset.copy()
    print(func.__name__)
    for i in tqdm(range(len(result['text']))):
        result['text'].iloc[i] = func(result['text'].iloc[i])
    return result


def remove_urls(dataset: pd.DataFrame) -> pd.DataFrame:
    """ Remove urls from tweets """
    def _remove_url(text: str) -> str:
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'', text)
    return _copy_and_apply(dataset, _remove_url)


def remove_html_tags(dataset: pd.DataFrame) -> pd.DataFrame:
    """ Remove html tags from tweets """
    def _remove_html_tags(text: str) -> str:
        html = re.compile(r'<.*?>')
        return html.sub(r'', text)
    return _copy_and_apply(dataset, _remove_html_tags)


def remove_emojis(dataset: pd.DataFrame) -> pd.DataFrame:
    """ Remove emojis from tweets """
    def _remove_emojis(text: str) -> str:
        emoji = re.compile("[" + "".join(EMOJIS) + "]+", flags=re.UNICODE)
        return emoji.sub(r'', text)
    return _copy_and_apply(dataset, _remove_emojis)


def remove_punctuation(dataset: pd.DataFrame) -> pd.DataFrame:
    """ Remove punctuation from tweets """
    def _remove_punctuation(text: str) -> str:
        table = str.maketrans('', '', string.punctuation)
        return text.translate(table)
    return _copy_and_apply(dataset, _remove_punctuation)


def correct_spelling(dataset: pd.DataFrame) -> pd.DataFrame:
    """ Correct spelling of tweets """
    checker = SpellChecker()
    def _correct_spelling(text: str) -> str:
        errors = checker.unknown(text.split())
        result = [checker.correction(w) if w in errors else w for w in text.split()]
        return " ".join(result)
    return _copy_and_apply(dataset, _correct_spelling)


def replace_words_not_covered_by_embeddings(dataset: pd.DataFrame) -> pd.DataFrame:
    """ Replace most frequent words not covered by embeddings with synonyms """
    def _replace_not_covered_words(text: str) -> str:
        text = text.lower()
        for word in text.split():
            if word in UNKNOWN_WORDS:
                text = text.replace(word, UNKNOWN_WORDS[word])
        return text
    return _copy_and_apply(dataset, _replace_not_covered_words)


def clean(dataset: pd.DataFrame) -> pd.DataFrame:
    """ Apply cleaning routine """
    dataset = remove_urls(dataset)
    dataset = remove_html_tags(dataset)
    dataset = remove_emojis(dataset)
    dataset = remove_punctuation(dataset)
    dataset = correct_spelling(dataset)
    dataset = replace_words_not_covered_by_embeddings(dataset)
    return dataset


if __name__ == "__main__":
    train, test = data.load()
    train_ = remove_urls(train.iloc[0:100])

    for i in range(100):
        print("@ i = " + str(i))
        print(train.iloc[i]['text'])
        print(train_.iloc[i]['text'])
