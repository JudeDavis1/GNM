import os
import torch
import warnings

from model import Trainer
from dataset import BookCorpusDataset

import logger


warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_built():
    device = torch.device('mps')

logger.INFO(f'Using {str(device).upper()} backend...')

def train():
    logger.INFO('Preparing data...')

    CHUNK_SIZE = 10
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
    trainer = Trainer(corpus)
    trainer.set_device(device)

    try:
        os.system('clear')
        trainer.fit_dataset(
            dataset,
            lr=0.0009,
            epochs=10,
            chunk_size=CHUNK_SIZE,
            batch_size=128,
            save_checkpoint=True,
        )
    except KeyboardInterrupt:
        trainer.save('GNM_model')

    trainer.plot_loss()


if __name__ == '__main__':
    train()

