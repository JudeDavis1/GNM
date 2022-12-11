
import torch
import numpy as np


from torch import nn
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import utils


'''

Contains all relevent torch.nn.Modules and the main
GENOME model (G.N.M. Generative Neural Model). This is a model that
can generate text based on samples of sentences.

'''


class GNMModel(nn.Module):

    '''
    Class:
        The main predictive model for GNM ("Generative Neural Model" or GENOME).
    Args:
        corpus:
            - The set of words that the model can understand.
            - Required to convert arbitrary text into indexes to
              be fed into the model.
        name:
            - The name of the file which saves the model parameters.
    '''


    def __init__(self, corpus, name='GNM_model'):
        super().__init__()

        # For plotting the model loss
        self.losses = []
        self.epochs = None

        self.corpus = corpus
        self.lstm_size = 128
        self.n_lstm_layers = 3
        self.name = name + '.pt'
        self.vocab_size = len(corpus)
        self.dim =  int(1.6 * np.sqrt(len(corpus)))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if torch.backends.mps.is_built():
            self.device = torch.device('mps')

        self.embedding = nn.Embedding(self.vocab_size, self.dim)
        self.attn_head = Attn(self.dim, self.lstm_size)
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.n_lstm_layers * 2,

            dropout=.6
        )
        self.fc = nn.Linear(self.lstm_size, self.vocab_size)
        self.dropout = nn.Dropout(.4)


    def forward(self, x: torch.Tensor, prev_state) -> torch.Tensor:
        # Get the 'word' vectors which are representations of an actual word index.
        embeddings = self.embedding(x.int()).float()
        attn = self.dropout(self.attn_head(embeddings))
        output, state = self.lstm(attn, prev_state)
        logits = self.dropout(self.fc(output))

        return logits, state


    def generate_tokens(self, x: str, chunk_size=10, training=False):

        '''
        Function:
            Generates a set of words to complete a phrase x: (see below).
        Args:
            x:
                - A starting phrase to complete.
            chunk_size: (default: 10)
                - The number of words to generate.
            training: (default: False)
                - Whether or not the model should generate in training mode.
        '''

        encoded_samples = x

        if not training:
            self.eval()
            self.cpu()
            encoded_samples = torch.tensor(np.array([utils.text2idx(x, self.corpus, autoadd=False)])).long().cpu()

        generated_tokens = []
        states = self.init_states(len(encoded_samples[0]), device='cpu')

        for i in range(chunk_size):
            output, states = self.forward(encoded_samples, states)
            probs = F.softmax(output[0][-1], dim=0).cpu().detach().numpy()
            idx = np.random.choice(len(output[0][-1]), p=probs)

            generated_tokens.append(idx)

        if not training:
            return ' '.join([self.corpus[idx] for idx in generated_tokens]).lstrip().rstrip()

        return torch.tensor(generated_tokens)


    def fit_dataset(self,
                    dataset: Dataset,
                    lr: float=0.01,
                    epochs: int=10,
                    batch_size: int=8,
                    chunk_size: int=5,
                    save_checkpoint: bool=True):

        '''
        Function:
            - Fits the model to a specified dataset (see Args below).
            - Begins by initializing training parameters (optimizer, loss function (criterion),
            and LSTM states) and begins training.
        Args:
            dataset:
                - A class which inherits from the Dataset class (see imports). This class
                is loaded into a DataLoader (see imports) which can split data into batches
                for training.
                - See (dataset.py) for template dataset.
            lr: (default: 0.01)
                - The "learning rate", which is the magnitude at which the optimizer should update
                the model parameters.
                - A large learning rate tends to over-shoot and a low learning rate tends to
                never converge.
                - See (https://www.jeremyjordan.me/nn-learning-rate/) for more info on
                setting the learning rate.
            epochs: (default: 10)
                - How many times the model should train itself on a set of batches
                (see batch_size).
            batch_size: (default: 8)
                - How many batches of data the model will look at in 1 epoch.
                - Lower batch sizes tends to take slightly longer but MAY lead to better
                results. Larger batch sizes may be quicker but the model may fail to generalize.
                IT DEPENDS ON THE TYPE OF PROBLEM.
            chunk_size: (default: 5)
                - The number of words the model should generate.
            save_checkpoint: (default: True)
                - If the model should save the model parameters for later use.
        '''

        print('Initializing training...')

        ld = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.99, 0.999))

        total_steps = len(dataset) // batch_size

        for i in (t := tqdm(range(1, epochs + 1))):
            '''
            states (h, c):
            h : hidden state.
            c : state at time: t.
            '''

            (h, c) = self.init_states(chunk_size)

            for j, (x, y) in enumerate(ld):
                optimizer.zero_grad()

                x = x.float().to(self.device)
                output, (h, c) = self.forward(x, (h, c))

                h = h.detach()
                c = c.detach()

                loss = criterion(output.transpose(1, 2), y.float().to(self.device).long()).float()

                loss.backward()

                # Convert to CPU so that CUDA doesn't run out of memory if using the Adam optimizer.
                # self.to(torch.device('cpu'), non_blocking=True)
                optimizer.step()
                # self.to(self.device, non_blocking=True)

                loss = loss.cpu().detach().numpy()
                t.set_description(f'Batch: {j + 1}/{total_steps} Loss: {loss:.3f}')

            self.epochs = i
            self.losses.append(loss)

        if save_checkpoint:
            self.cpu().save(self.name)


    def save(self, path: str):
        torch.save(self.state_dict(), path)


    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))


    def plot_loss(self):
        plt.plot(range(1, self.epochs + 1), self.losses)
        plt.show()


    def init_states(self, seq_length, device=None) -> tuple:
        zero_state = torch.zeros(
            self.n_lstm_layers * 2,
            seq_length,
            self.lstm_size,
            device=device if device else str(self.device)
        )

        return (zero_state, zero_state)



class Attn(nn.Module):

    '''
    Class:
        Outputs probability distributions multiplied with embeddings.
        The Attn layer computes the importance of a word based on a softmax function.
    '''

    def __init__(self, n_in, n_out):
        super().__init__()

        self._in  = n_in
        self._out = n_out
        self.fc   = nn.Linear(n_in, 1, bias=False)
        self.ext1  = nn.Linear(n_in, n_out, bias=False)

    def forward(self, embeds: torch.Tensor):
        weights = self.fc(embeds)
        weights_norm = F.softmax(weights).float()

        attn_probs = torch.multiply(weights_norm, embeds)
        ext: torch.Tensor = self.ext1(attn_probs)

        return ext

    @torch.no_grad()
    def _init_weights(self, m):
        m.weight.fill_(1.0)
        torch.var(m.weight, False)




