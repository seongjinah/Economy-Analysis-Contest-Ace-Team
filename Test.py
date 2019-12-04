from konlpy.tag import Okt
from keras import optimizers
from keras import metrics
from keras import models
from keras import layers
from keras import losses

import numpy as np
import json


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

with open('selected_words.json', encoding='utf-8', mode='r') as f:
    selected_words = json.load(f)

with open('train_x.json', encoding='utf-8', mode='r') as f:
    train_x = json.load(f)

with open('train_y.json', encoding='utf-8', mode='r') as f:
    train_y = json.load(f)

x_train = np.asarray(train_x).astype('float32')
y_train = np.asarray(train_y).astype('float32')

model = models.Sequential()

model.add(layers.Dense(64, activation='relu', input_shape=(1000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

model.fit(x_train, y_train, epochs=20, batch_size=512)

testdata = read_data('test.csv')

f1 = open('submission.csv', encoding='utf-8', mode='w')
f1.write('id,smishing' + '\n')

for td in testdata:
    review = td[2]
    t = tokenize(review)
    tf = term_frequency(t)
    data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
    score = float(model.predict(data))
    if score > 0.5:
        f1.write(td[0] + ',' + str(score * 100) + '\n')
    else:
        f1.write(td[0] + ',' + str(score * 100) + '\n')

