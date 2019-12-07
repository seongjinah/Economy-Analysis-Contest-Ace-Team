from keras.layers import SimpleRNN, Embedding, Dense
from keras.models import Sequential
from konlpy.tag import Okt
import matplotlib.pyplot as plt
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

with open('stopwords/selected_words.json', encoding='utf-8', mode='r') as f:
    selected_words = json.load(f)

with open('stopwords/train_x.json', encoding='utf-8', mode='r') as f:
    x_data = json.load(f)

with open('stopwords/train_y.json', encoding='utf-8', mode='r') as f:
    y_data = json.load(f)

n_of_train = int(len(x_data) * 0.8)
n_of_test = int(len(x_data) - n_of_train)

test_x = x_data[n_of_train:]
test_y = y_data[n_of_train:]
train_x = x_data[:n_of_train]
train_y = y_data[:n_of_train]

x_test = np.asarray(test_x).astype('float32')
y_test = np.asarray(test_y).astype('float32')
x_train = np.asarray(train_x).astype('float32')
y_train = np.asarray(train_y).astype('float32')

vocab_size = len(selected_words) + 1

model = Sequential()
model.add(Embedding(vocab_size), 32)
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

print("\n 테스트 정확도: %.4f" % (model.evaluate(x_test, y_test)[1]))

epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()