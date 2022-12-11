import warnings
warnings.filterwarnings('ignore')

import io
import torch

from typing import List

from model import GNMModel


def main():
    corpus: List[str] = [word.strip('\n') for word in io.open('corpus.txt', encoding='utf-8').readlines()]

    model = GNMModel(corpus).cpu()
    model.load('GNM_model')

    while True:
        text = input('Text > ')

        print(model.generate_tokens(text))


if __name__ == '__main__':
    main()

