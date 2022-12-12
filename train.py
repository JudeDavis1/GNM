import os
import sys
import torch
import warnings

from model import Runner
from dataset import BookCorpusDataset

import logger


if len(sys.argv) != 2:
    logger.CRITICAL('Please supply the number of epochs.')
    exit(1)

warnings.filterwarnings('ignore')
CHUNK_SIZE = 1
EPOCHS = int(sys.argv[1])

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_built():
        device = torch.device('mps')

    logger.INFO(f'Using {str(device).upper()} backend...')
    logger.INFO('Preparing data...')
    dataset = BookCorpusDataset(
        CHUNK_SIZE,
        corpus_from_file='corpus.txt',
        save_train_data=True,
        train_data_file='train_data.csv.gz'
    )

    dataset.n_batches = 5000
    logger.INFO('Generating batches...')
    dataset.generate_batches()
    logger.INFO(f'Generated {dataset.n_batches} batches')

    corpus = dataset.corpus
    trainer = Runner(corpus)
    trainer.set_device(device)
    trainer.load('GNM_model')

    try:
        os.system('clear')
        trainer.fit_dataset(
            dataset,
            lr=0.0009,
            epochs=EPOCHS,
            chunk_size=CHUNK_SIZE,
            batch_size=256,
            save_checkpoint=True,
        )
    except KeyboardInterrupt:
        trainer.save('GNM_model')

    trainer.plot_loss()


if __name__ == '__main__':
    train()

