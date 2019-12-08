import os
import json
import numpy as np
import pandas as pd
from konlpy.tag import Okt
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def read_data(filename):
    with open(filename, encoding='utf-8', mode='r') as f:
        data = [line.split(',') for line in f.read().splitlines()]
        data = data[1:]
    return data


okt = Okt()


#   data['smishing'].value_counts().plot(kind='bar')
#   plt.show()

#   print(data.groupby('smishing').size().reset_index(name='count'))


okt = Okt()

stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다',
             'XXX', '.', '을', '-', '(', ')', ':', '!', '?', ')-', '.-', 'ㅡ', 'XXXXXX', '..', '.(']

x_train = []

if os.path.isfile('act1_1.json'):
    with open('act1_1.json', encoding='utf-8', mode='r') as f:
        x_train = json.load(f)
else:
    data = pd.read_csv('train.csv', encoding='utf-8')

    del data['id']
    del data['year_month']

    data['text'] = data['text'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")

    for sentence in data['text']:
        temp_x = []
        temp_x = okt.morphs(sentence, stem=True)
        for word in temp_x:
            if word in stopwords:
                continue
            else:
                x_train.append(temp_x)

    with open('act1_1.json', encoding="utf-8", mode='w') as make_file:
        json.dump(x_train, make_file, ensure_ascii=False, indent="\t")

print(x_train[:3])

max_words = 35000
tokenizer = Tokenizer(num_words=max_words) # 상위 35,000개의 단어만 보존
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)

print('문자의 최대 길이 :', max(len(l) for l in x_train))
print('문자의 평균 길이 :', sum(map(len, x_train))/len(x_train))

plt.hist([len(s) for s in x_train], bins=50)
plt.xlabel('length of Data')
plt.ylabel('number of Data')
plt.show()

max_len = 30
# 전체 데이터의 길이는 30으로 맞춘다.

x_train = pad_sequences(x_train, maxlen=max_len)
y_train = np.array(data['smishing'])
