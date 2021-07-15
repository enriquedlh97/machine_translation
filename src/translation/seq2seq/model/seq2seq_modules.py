""" Main seq2seq model modules

Created: July 15, 2021
Updated: July 15, 2021

Description:
    Implements the Encoder and Decoder modules used for the seq2seq model
"""
import torch.nn as nn


class Encoder(nn.Module):
    """

    """

    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        """

        :param input_size:
        :param embedding_size:
        :param hidden_size:
        :param num_layers:
        :param dropout:
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding)

        return hidden, cell


class Decoder(nn.Module):
    """

    """

    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout):
        """

        :param input_size:
        :param embedding_size:
        :param hidden_size:
        :param output_size:
        :param num_layers:
        :param dropout:
        """
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        """

        :param x:
        :param hidden:
        :param cell:
        :return:
        """

        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))

        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))

        predictions = self.fc(outputs)

        predictions = predictions.squeeze(0)

        return predictions, hidden, cell
