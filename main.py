from konlpy.tag import Kkma
from konlpy.utils import pprint


def read_data(filename):
    t = open(filename, encoding='utf-8', mode='r')
    t = t.read()
    return t


train_data = read_data('train.csv')
test_data = read_data('test.csv')


kkma = Kkma()

data = open('train.csv', encoding='utf-8', mode='r')
data0 = open('text0.csv', encoding='utf-8', mode='w')
data1 = open('text1.csv', encoding='utf-8', mode='w')

a = data.read()
a = a.split('\n')

for b in a:
    c = b.split(',')

    if len(c) <= 3:
        continue

    if c[3] == '0':
        data0.write(c[2] + '\n')

    elif c[3] == '1':
        data1.write(c[2] + '\n')
