'''

The main Dataset class for the entire model is written here.
It keeps track of train_data and the corpus and everything to do
with the book dataset. Also supports multithreading.

'''


import io
import re
import os
import time
import torch
import utils
import string
import numpy as np

from typing import List, Optional, Tuple
from torch.utils.data import Dataset


from multiprocessing import Pool
from nltk.tokenize import word_tokenize

import utils
import logger



allowed_punc = '?.!-'

class BookCorpusDataset(Dataset):

    '''
    Class:
        - The main dataset which contains all the required data for training
        the model.
        - Supports multiprocessing.
    Args:
        chunk_size:
            - The amount of words in a batch.
            - This is set to None, when just_corpus=True.
        threads:
            - The number of threads to use when multiprocessing text.
        just_corpus:
            - Whether the dataset should only prepare the corpus (when not training).
            - You can't run generate_batches() if this is set to True.
        save_corpus:
            - Whether to save the corpus in a file or not.
        save_train_data:
            - Whether or not to save the training data instead of processing it every time
            at runtime.
        train_data_file:
            - The filename to load the training data from.
        corpus_from_file:
            - The filename to load the corpus from.
    '''

    def __init__(self,
                 chunk_size=3,
                 threads=2,
                 just_corpus=False,
                 save_corpus=True,
                 save_train_data=False,
                 train_data_file: Optional[str]=None,
                 corpus_from_file: Optional[str]=None):
        global m_pool
        m_pool = Pool(threads if threads else 1)
        print('Preparing data...')

        try:
            assert bool(train_data_file) == bool(corpus_from_file)
        except AssertionError:
            raise ValueError('''If train_data_file is None, then so should the corpus_from_file.
            corpus_from_file is dependant on train_data_file.''')

        start = time.time()
        self.n_threads = threads
        self.n_batches = 500
        self._just_corpus = just_corpus
        self.chunk_size = chunk_size if not self._just_corpus else None

        if just_corpus:
            return

        file_contents: List[str]

        if corpus_from_file:
            self.corpus = [word.strip('\n') for word in io.open(corpus_from_file, encoding='utf-8').readlines()]
            file_contents = load_corpus('data', threads=threads, just_contents=True)
        else:
            self.corpus, file_contents = load_corpus('data', threads=threads)

        if train_data_file:
            self.train_data = np.loadtxt(train_data_file)
        else:
            r = m_pool.map(utils.preprocess_text, file_contents.split(' '))

            self.train_data_str = r
            self.train_data: np.ndarray = utils.text2idx(self.train_data_str, self.corpus, preprocessed=True)

            print(f'Process took: {time.time() - start}')

            if save_corpus:
                with io.open('corpus.txt', 'w', encoding='utf-8') as f:
                    for word in self.corpus:
                        f.write(word + '\n')

            if save_train_data:
                np.savetxt('train_data.csv.gz', self.train_data)

        self.prep_data = []

        self._n_threads = threads


    def generate_batches(self):
        try:
            assert not self._just_corpus
        except AssertionError:
            raise AssertionError('If you want to run: generate_batches(), you must set (just_corpus = False) in the constructor.')

        beginning = 0
        last_idx = self.chunk_size

        if self.n_threads:
            for i in range(self.n_batches):
                # Reuse the pool for generating batches
                self.prep_data.append(m_pool.apply(self._get_batch, args=(self, beginning, last_idx)))

                beginning = last_idx
                last_idx += self.chunk_size

            m_pool.close()
            m_pool.join()
        else:
            # If 0 threads are required
            for i in range(self.n_batches):
                # Reuse the pool for generating batches
                self.prep_data.append(self._get_batch(self, beginning, last_idx))

                beginning = last_idx
                last_idx += self.chunk_size

    @staticmethod
    def _get_batch(self, beginning, last_idx):
        starting_phrase = self.train_data[beginning:last_idx]
        target_word = self.train_data[last_idx:last_idx + self.chunk_size]

        return (starting_phrase, target_word)

    def __getitem__(self, index):
        return self.prep_data[index]

    def __len__(self):
        return len(self.prep_data)



def load_corpus(text_file_dir, threads=6, just_contents=False) -> Tuple[list, str]:
    files = [open('data/' + f, 'r', encoding='utf-8') for f in os.listdir(text_file_dir)]

    corpus = ''

    logger.INFO('Collecting tokens from:\n')
    files_str = os.listdir(text_file_dir)
    files_str.sort(key=len)

    for c in files_str:
        logger.info(c)
    print()

    for f in files:
        corpus += f.read()

    total_text = corpus

    r = []
    if not just_contents:
        edited_corpus = re.sub("[+]", '', corpus)
        edited_corpus = re.sub(f"(CHAPTER \w+.\s)", '', corpus)
        edited_corpus = re.sub(f"\n", ' ', corpus)
        words = edited_corpus.split(' ')

        if threads:
            # Multiprocessing pool
            m_pool = Pool(threads)

            # filter punctuation and stop-words
            r = m_pool.map(proc, words)
            m_pool.close()
            m_pool.join()
        else:
            r = [proc(word) for word in words]

        return (list(set(r)), total_text)
    return total_text


def proc(word):
    new_word = ''

    for char in word:
        if char in allowed_punc or char in string.ascii_letters:
            new_word += char

    if word != '':
        new_word = new_word.strip()
    return new_word

