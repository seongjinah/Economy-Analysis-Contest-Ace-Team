from konlpy.tag import Okt
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

tokens = list()
clean_token = list()
clean_tokens = list()

train_data = read_data('train.csv')

if os.path.isfile('okt.json'):
    with open('okt.json', encoding='utf-8', mode='r') as f:
        token = json.load(f)
else:
    token = ((tokenize(row[2]), row[3]) for row in train_data)
    with open('okt.json', encoding="utf-8", mode='w') as make_file:
        json.dump(token, make_file, ensure_ascii=False, indent="\t")

for row in token:
    for data in row[0]:
        tokens.append([data, row[1]])

stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다',
             'XXX', '.', '을', '-', '(', ')', ':', '!', '?', ')-', '.-', 'ㅡ', 'XXXXXX', '..', '.(']
stopwordset = ['Number']

for t in tokens:
    tex = t[0]
    num = t[1]
    t1, t2 = tex.split('/')
    if t1 in stopwords:
        continue
    if t2 in stopwordset:
        continue
    clean_tokens.append(t1)
    clean_token.append(t)

text = nltk.Text(clean_tokens, name='NMSC')

if os.path.isfile('selected_words.json'):
    with open('selected_words.json', encoding='utf-8', mode='r') as f:
        selected_words = json.load(f)
else:
    selected_words = [f[0] for f in text.vocab().most_common(10000)]
    with open('selected_words.json', encoding="utf-8", mode='w') as make_file:
        json.dump(selected_words, make_file, ensure_ascii=False, indent="\t")

if os.path.isfile('train_x.json'):
    with open('train_x.json', encoding='utf-8', mode='r') as f:
        train_x = json.load(f)
else:
    train_x = [term_frequency(d[0]) for d in clean_token]
    with open('train_x.json', encoding="utf-8", mode='w') as make_file:
        json.dump(train_x, make_file, ensure_ascii=False, indent="\t")

if os.path.isfile('train_y.json'):
    with open('train_y.json', encoding='utf-8', mode='r') as f:
        train_y = json.load(f)
else:
    train_y = [d[1] for d in clean_token]
    with open('train_y.json', encoding="utf-8", mode='w') as make_file:
        json.dump(train_y, make_file, ensure_ascii=False, indent="\t")
