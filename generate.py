import io
import warnings
warnings.filterwarnings('ignore')

from typing import List

from model import Runner


def main():
    corpus: List[str] = [word.strip('\n') for word in io.open('corpus.txt', encoding='utf-8').readlines()]

    model = Runner(corpus)
    model.load('GNM_model')

    while True:
        text = input('Text > ')

        print(model.generate_tokens(text))


if __name__ == '__main__':
    main()

