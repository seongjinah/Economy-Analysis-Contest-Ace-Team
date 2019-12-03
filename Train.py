from matplotlib import font_manager, rc
from konlpy.tag import Okt
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import json
import os
import nltk

okt = Okt()


# read_data(filename) : 데이터 불러오는 함수
def read_data(filename):
    with open(filename, encoding='utf-8', mode='r') as f:
        data = [line.split(',') for line in f.read().splitlines()]
        data = data[1:]
    return data


# tokenize(doc) : Okt 를 사용해 말뭉치 분석과 태그를 달아주는 함수
def tokenize(doc):
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]


train_data = read_data('train.csv')

print('file uploading . . .')

# token : train.csv list. [[태그달린 단어], [스미싱 여부]]

# train_token.json 파일이 있을때
if os.path.isfile('train_token.json'):
    with open('train_token.json', encoding='utf-8', mode='r') as f:
        token = json.load(f)

# train_token.json 파일이 없을때
else:
    print('making json file . . .')

    token = [(tokenize(row[2]), row[3]) for row in train_data]

    with open('train_token.json', mode='w', encoding="utf-8") as make_file:
        json.dump(token, make_file, ensure_ascii=False, indent="\t")

print('tokenizing . . .')

# tokens : 태그달린 단어만 모아둔 리스트
tokens = [t for data in token for t in data[0][0]]

# clean_tokens : 필요없는 단어들을 제거한 리스트
clean_tokens = list()

# tokens 걸러주기
for t in tokens:
    t1, t2 = t.split('/')
    # 현재 clean_tokens 조건 -> 이후 stopwords 배열에 넣어 작동시킬 예정
    # Yes : Noun, Verb
    # No : '하다', '되다'
    if t1 == '하다' or t1 == '되다':
        continue
    if t2 == 'Noun' or t2 == 'Verb':
        clean_tokens.append(t)

text = nltk.Text(clean_tokens, name='NMSC')

print('number of all tokens :')
print(len(text.tokens))

print('number of all token except same things: ')
print(len(set(text.tokens)))

print('most common token lists :')
pprint(text.vocab().most_common(10))

# 자주 나오는 단어 50개 그래프로 나타내기
# 폰트 필요 !

font_fname = 'C:/Windows/Fonts/gulim.ttc'
font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font', family=font_name)

plt.figure(figsize=(20, 10))
text.plot(50)
plt.show()
