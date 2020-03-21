#!/usr/bin/env python3

import numpy as np
from typing import Tuple


GLOVE_6B_PATH = "../glove/glove_6b/glove.6B.50d.txt"


def glove_6b_dict():
    """ Return dict with GloVe 6B embeddings """
    def line_to_embedding(line: str) -> Tuple[str, np.ndarray]:
        tokens = line.split()
        return (tokens[0], np.asarray(tokens[1:], 'float32'))
    with open(GLOVE_6B_PATH, 'r') as f:
        embeddings = (line_to_embedding(l) for l in f)
        return {k: v for k, v in embeddings}


if __name__ == "__main__":
    glove = glove_6b_dict()
    print(len(glove))
