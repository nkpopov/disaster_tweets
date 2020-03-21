#!/usr/bin/env python3

import nltk
nltk.download("wordnet")

import numpy as np
from nltk.corpus import wordnet
from typing import List


def get_synonyms(word: str) -> List[str]:
    """ Get list of synonyms for the provided word """
    synonyms = {l.name().lower() for l in s for s in wordnet.synsets(word)}
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


def replace_synonyms(text: str, n: int) -> str:
    """ Replace n random words in text with random synonyms """
    words = text.split()
    indices = [int(v * (len(words) - 1)) for v in np.random.rand(n)]
    

if __name__ == "__main__":
    synonyms = get_synonyms("disaster")
    print(synonyms)
