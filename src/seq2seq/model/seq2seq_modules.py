""" Text preprocessing for seq2seq model

Created: July 15, 2021
Updated: July 15, 2021

Description:
"""
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy

spacy_ger = spacy.load('de')
spacy_eng = spacy.load('en')


def tokenizer_german(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenizer_english(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


def generate_dataset():
    spacy_ger = spacy.load('de')
    spacy_eng = spacy.load('en')

    german = Field(tokenize=tokenizer_german, lower=True, init_token='<sos>', eos_token='<eos>')
    english = Field(tokenize=tokenizer_english, lower=True, init_token='<sos>', eos_token='<eos>')

    train_data, validation_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(german, english))

    german.build_vocab(train_data, max_size=10000, min_freq=2)
    english.build_vocab(train_data, max_size=10000, min_freq=2)


#
german = Field(tokenize=tokenizer_german, lower=True, init_token='<sos>', eos_token='<eos>')
english = Field(tokenize=tokenizer_english, lower=True, init_token='<sos>', eos_token='<eos>')


train_data, validation_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(german, english))

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)
