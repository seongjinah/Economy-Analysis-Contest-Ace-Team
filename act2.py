from konlpy.tag import Okt
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import numpy as np
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


with open('selected_words.json', encoding='utf-8', mode='r') as f:
    selected_words = json.load(f)


def term_frequency(doc):
    return [doc.count(word) for word in selected_words]


with open('train_x.json', encoding='utf-8', mode='r') as f:
    train_x = json.load(f)
with open('train_y.json', encoding='utf-8', mode='r') as f:
    train_y = json.load(f)
with open('test_x.json', encoding='utf-8', mode='r') as f:
    test_x = json.load(f)
with open('test_y.json', encoding='utf-8', mode='r') as f:
    test_y = json.load(f)

x_train = np.asarray(train_x).astype('float32')
y_train = np.asarray(train_y).astype('float32')
x_test = np.asarray(test_x).astype('float32')
y_test = np.asarray(test_y).astype('float32')

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
             loss=losses.binary_crossentropy,
             metrics=[metrics.binary_accuracy])

model.fit(x_train, y_train, epochs=10, batch_size=512)
results = model.evaluate(x_test,y_test)
print(results)

testdata = read_data('test.csv')
f1 = open('result_b.csv', encoding='utf-8', mode='w')
f1.write('id,smishing'+'\n')
ret = 0
cnt = 0

for td in testdata:
    review = td[2]
    t = tokenize(review)
    tf = term_frequency(t)
    data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
    score = float(model.predict(data))
    f1.write(td[0]+','+str(score*100)+'\n')
