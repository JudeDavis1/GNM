import string
import numpy as np

from tqdm import tqdm
from typing import Union
from multiprocessing import Pool

'''

Just basic utilities relating to encoding data and
other lexical word processing tools.

'''


def preprocess_text(text: str) -> str:
    new_text = ''

    for char in text:
        is_sep = char in [' ', '\n']

        if char.isalpha() or is_sep:
            new_text += char

    new_text = ' '.join([word.strip()
        if ' ' in word and word else word
        for word in new_text.split(' ')
    ]).lstrip().rstrip()

    return new_text.lower()

def text2idx(
    text: Union[str, list],
    corpus: list,
    preprocessed: bool=False,
    autoadd: bool=True,
    pbar: bool=False
) -> np.ndarray:
    words = text
    encoded = []
    should_preprocess = False

    if type(text) == str:
        # split by word
        if not preprocessed:
            should_preprocess = True
            words = words.split(' ')
    elif type(text) in [list]:
        if not preprocessed:
            should_preprocess = True

    i = 0
    iter = tqdm(words) if pbar else words
    for word in iter:
        if should_preprocess:
            word = preprocess_text(word)

        if autoadd and word not in corpus:
            corpus.append(word)

        encoded.append(corpus.index(word))

        if i % 1000 == 0 and type(iter) == tqdm:
            iter.update(1000)

        i += 1

    return np.array(encoded, dtype=np.int)

def idx2text(idxs: Union[list, np.ndarray], corpus: list):
    words = [corpus[idx].strip() for idx in idxs]
    return ' '.join(words).lstrip().rstrip()

def rm_empty_strings(l: list) -> list:
    return [chunk for chunk in l if chunk]
