import os
import sys
import torch
import asyncio
import warnings

from model import GNMModel
from dataset import BookCorpusDataset

import logger


warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_built():
    device = torch.device('mps')

logger.INFO(f'Using {device} backend...')

def train():
    logger.INFO('Preparing data...')

    CHUNK_SIZE = 10
    dataset = BookCorpusDataset(
        CHUNK_SIZE,
        corpus_from_file=None,
        save_train_data=True,
        train_data_file=None
    )

    dataset.n_batches = 10
    logger.INFO('Generating batches...')
    dataset.generate_batches()
    logger.INFO(f'Generated {dataset.n_batches} batches')

    print(dataset.train_data_str)

    corpus = dataset.corpus
    model = GNMModel(corpus).to(device)

    # model.load('GNM_model')

    try:
        os.system('clear')
        model.fit_dataset(
            dataset,
            lr=0.0009,
            epochs=100,
            chunk_size=CHUNK_SIZE,
            batch_size=64,
            save_checkpoint=True,
        )
    except KeyboardInterrupt:
        model.save('GNM_model')

    model.plot_loss()


if __name__ == '__main__':
    train()

