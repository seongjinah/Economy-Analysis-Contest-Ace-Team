from konlpy.tag import Okt
from pprint import pprint

import json
import os
import nltk

okt = Okt()


def read_data(filename):
    with open(filename, encoding='utf-8', mode='r') as f:
        data = [line.split(',') for line in f.read().splitlines()]
        data = data[1:]
    return data


def tokenize(doc):
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]


train_data = read_data('train.csv')
test_data = read_data('test.csv')


if os.path.isfile('train_token.json'):
    with open('train_token.json', encoding='utf-8', mode='r') as f:
        token = json.load(f)

else:
    token = [(tokenize(row[2]), row[3]) for row in train_data]

    with open('train_token.json', mode='w', encoding="utf-8") as make_file:
        json.dump(token, make_file, ensure_ascii=False, indent="\t")
# d = 0
tokens = [t for d in token for t in d[0][0]]

text = nltk.Text(tokens, name='NMSC')

# 전체 토큰의 개수
print(len(text.tokens))

# 중복을 제외한 토큰의 개수
print(len(set(text.tokens)))

# 출현 빈도가 높은 상위 토큰 10개
pprint(text.vocab().most_common(10))
