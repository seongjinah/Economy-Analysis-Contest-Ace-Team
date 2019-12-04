from konlpy.tag import Okt
import numpy as np
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


def term_frequency(doc):
    return [doc.count(word) for word in selected_words]


okt = Okt()

print('start reading train.csv')

train_data = read_data('train.csv')
if os.path.isfile('train_token.json'):
    with open('train_token.json', encoding='utf-8', mode='r') as f:
        token = json.load(f)
else:
    print('start making train_token.json')
    token = [(tokenize(row[2]), row[3]) for row in train_data]
    with open('train_token.json', encoding="utf-8", mode='w') as make_file:
        json.dump(token, make_file, ensure_ascii=False, indent="\t")
        print('end making train_token.json')

print('end reading train.csv')

print('start making tokens list')

tokens = [t for d in token for t in d[0][0]]

print('end making tokens list')

print('start making clean_tokens list')

clean_tokens = list()

stopwords = ['XXX', '하다', '되다']
stopwordset = ['Josa', 'Punctuation', 'Number', 'Suffix', 'Modifier']

# tokens 걸러주기
for t in tokens:
    t1, t2 = t.split('/')
    if t1 in stopwords:
        continue
    if t2 in stopwordset:
        continue
    clean_tokens.append(t)

print('end making clean_tokens list')

print('start making selected_words.json')

text = nltk.Text(clean_tokens, name='NMSC')
if os.path.isfile('selected_words.json'):
    with open('selected_words.json', encoding='utf-8', mode='r') as f:
        selected_words = json.load(f)
else:
    selected_words = [f[0] for f in text.vocab().most_common(1000)]
    with open('selected_words.json', encoding="utf-8", mode='w') as make_file:
        json.dump(selected_words, make_file, ensure_ascii=False, indent="\t")

print('end making selected_words.json')

print('start making train_x.json')

if os.path.isfile('train_x.json'):
    with open('train_x.json', encoding='utf-8', mode='r') as f:
        train_x = json.load(f)
else:
    train_x = [term_frequency(d[0][0]) for d in token]
    with open('train_x.json', encoding="utf-8", mode='w') as make_file:
        json.dump(train_x, make_file, ensure_ascii=False, indent="\t")

print('end making train_x.json')

print('start making train_y.json')

if os.path.isfile('train_y.json'):
    with open('train_y.json', encoding='utf-8', mode='r') as f:
        train_y = json.load(f)
else:
    train_y = [d[0][1] for d in token]
    with open('train_y.json', encoding="utf-8", mode='w') as make_file:
        json.dump(train_y, make_file, ensure_ascii=False, indent="\t")

print('end making train_y.json')

x_train = np.asarray(train_x).astype('float32')
y_train = np.asarray(train_y).astype('float32')

print('end Preprocessing')
