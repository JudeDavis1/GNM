'''

The main Dataset class for the entire model is written here.
It keeps track of train_data and the corpus and everything to do
with the book dataset. Also supports multithreading.

'''


import io
import re
import os
import time
import utils
import string
import asyncio
import warnings
import numpy as np

from torch.utils.data import Dataset
from typing import List, Optional, Tuple, Union


from multiprocessing import Pool

warnings.filterwarnings('ignore')

import utils
import logger


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
                 just_corpus=False,
                 save_corpus=True,
                 save_train_data=False,
                 train_data_file: Optional[str]=None,
                 corpus_from_file: Optional[str]=None):
        print('Preparing data...')

        try:
            assert bool(train_data_file) == bool(corpus_from_file)
        except AssertionError:
            raise ValueError('''If train_data_file is None, then so should the corpus_from_file.
            corpus_from_file is dependant on train_data_file.''')

        start = time.time()
        self.n_batches = 500
        self._just_corpus = just_corpus
        self.loop = asyncio.get_event_loop()
        self.chunk_size = chunk_size if not self._just_corpus else None

        if just_corpus:
            return

        file_contents: List[str]

        if corpus_from_file:
            # Remove newlines
            self.corpus = [word.strip('\n') for word in io.open(corpus_from_file, encoding='utf-8').readlines()]
            file_contents = self._run_load_corpus(just_contents=True)
        else:
            self.corpus, file_contents = self._run_load_corpus()

        if train_data_file:
            self.train_data = np.loadtxt(train_data_file)
        else:
            logger.INFO('Preprocessing...')
            self.train_data: np.ndarray = utils.text2idx(
                file_contents,
                self.corpus,
                preprocessed=True,
                pbar=True
            )
            logger.INFO('Finished preprocesing')

            print(f'Process took: {time.time() - start}')

            if save_corpus:
                with io.open('corpus.txt', 'w', encoding='utf-8') as f:
                    for word in self.corpus:
                        f.write(word + '\n')

            if save_train_data:
                np.savetxt('train_data.csv.gz', self.train_data)

        self.prep_data = []

    def generate_batches(self):
        try:
            assert not self._just_corpus
        except AssertionError:
            raise AssertionError('If you want to run: generate_batches(), you must set (just_corpus = False) in the constructor.')

        beginning = 0
        last_idx = self.chunk_size

        for i in range(self.n_batches):
            sample = self._get_batch(beginning, last_idx)
            self.prep_data.append(sample)

            beginning = last_idx
            last_idx += self.chunk_size

    def _run_load_corpus(self, just_contents=False):
        return self.loop.run_until_complete(load_corpus('data', just_contents=just_contents))

    def _get_batch(self, beginning, last_idx):
        starting_phrase = self.train_data[beginning:last_idx]
        target_word = self.train_data[last_idx:last_idx + self.chunk_size]

        return (starting_phrase, target_word)

    def __getitem__(self, index):
        return self.prep_data[index]

    def __len__(self):
        return len(self.prep_data)



async def load_corpus(text_file_dir, just_contents=False) -> Union[list, str]:
    corpus = ''
    files_str = os.listdir(text_file_dir)
    files = [open('data/' + f, 'r', encoding='utf-8') for f in files_str]

    logger.INFO('Collecting tokens from:\n')
    files_str.sort(key=len)

    for c in files_str:
        logger.info(c)
    print()

    for f in files:
        corpus += f.read()
    total_text = corpus

    if not just_contents:
        edited_corpus = re.sub(f"\n", ' ', corpus)
        words = edited_corpus.split(' ')

        # Filter punctuation and stop-words
        r = []
        for word in words:
            r.append(await proc(word))

        return (list(set(r)), total_text)
    return total_text


async def proc(word):
    new_word = ''
    allowed_punc = '?.!-'

    for char in word:
        if char in allowed_punc or char in string.ascii_letters:
            new_word += char

    if word != '':
        new_word = new_word.strip()
    return new_word


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    start = time.time()
    corpus, text = loop.run_until_complete(load_corpus('data'))
    end = time.time()
    print(end - start)
