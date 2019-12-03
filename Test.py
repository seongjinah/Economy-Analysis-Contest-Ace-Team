from keras import optimizers
from keras import models
from keras import layers
from keras import losses
from keras import metrics
from konlpy.tag import Okt
from pprint import pprint

import numpy as np
import tensorflow
import json
import os
import nltk


def read_data(filename):
    with open(filename, encoding='utf-8', mode='r') as f:
        data = [line.split(',') for line in f.read().splitlines()]
        data = data[1:]
    return data


def tokenize(doc):
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]


okt = Okt()

test_data = read_data('test.csv')

print('file uploading . . .')

if os.path.isfile('test_token.json'):
    with open('test_token.json', encoding='utf-8', mode='r') as f:
        token = json.load(f)

else:
    print('making json file . . .')

    token = [tokenize(row[2]) for row in test_data]

    with open('test_token.json', mode='w', encoding="utf-8") as make_file:
        json.dump(token, make_file, ensure_ascii=False, indent="\t")

print('tokenizing . . .')

tokens = [t for data in token for t in data]

text = nltk.Text(tokens, name='NMSC')

print('number of all tokens :')
print(len(text.tokens))

print('number of all token except same things: ')
print(len(set(text.tokens)))

print('most common token lists :')
pprint(text.vocab().most_common(10))