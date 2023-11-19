import random

TESTNUM = 100000
KEYRANGE = 1000
VALRANGE = 10.0

f = open("seq_impl/in.txt", "w")



for i in range(TESTNUM):
    key = random.randrange(0, KEYRANGE, 1)
    value = random.uniform(-VALRANGE, VALRANGE)
    f.write(f'{key} {value:.3f}\n')

f.close()