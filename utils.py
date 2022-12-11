import string
import numpy as np

from typing import Union
from multiprocessing import Pool

'''

Just basic utilities relating to encoding data and 
other lexical analysis/word processing tools.

'''



def preprocess_text(text: str) -> str:
    new_text = ''

    for char in text:
        is_sep = char in [' ', '\n']

        if char.isalpha() or is_sep:
            new_text += char
    
    new_text = ' '.join([word.strip() if ' ' in word and word else word for word in new_text.split(' ')]).rstrip().lstrip()

    return new_text.lower()

def text2idx(text: Union[str, list], corpus: list, preprocessed=False, autoadd=True) -> np.ndarray:
    words = []  # just to declare (stopping the Unbound error)
    encoded = []

    if type(text) == str:
        # split by word
        words = preprocess_text(text).split(' ')
    elif type(text) == list:
        words = text
        if not preprocessed:
            words = [preprocess_text(chunk) for chunk in text]

    for word in words:
        if not word in corpus:
            if autoadd:
                corpus.append(word)
        
        encoded.append(corpus.index(word))
    
    return np.array(encoded, dtype=np.int)

def idx2text(idxs: Union[list, np.ndarray], corpus: list):
    words = []

    for i in range(len(idxs)):
        words.append(corpus[idxs])
    
    return ' '.join(words).lstrip().rstrip()

def rm_empty_strings(l: list) -> list:
    return [chunk for chunk in l if chunk]
