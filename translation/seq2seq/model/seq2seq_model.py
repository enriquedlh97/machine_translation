""" Main seq2seq model

Created: July 15, 2021
Updated: July 15, 2021

Description:
    Implements the seq2seq model
"""
import torch
import torch.nn as nn
import random


class Seq2Seq(nn.Module):
    """

    """

    def __init__(self, encoder, decoder):
        """

        :param encoder:
        :param decoder:
        """
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, english, device, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        hidden, cell = self.encoder(source)

        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)

            outputs[t] = output

            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs
