import random

TESTNUM = 1000000
# TESTNUM = 10
KEYRANGE = 100
VALRANGE = 10

f = open("inputs/in.txt", "w")

# in the future, to let each key's value have a 
# distribution centered at 0, (so that we don't overflow)
# we can generate a random size for each key, and then generate a sequence. then shuffle before the next key value.


for i in range(TESTNUM):
    key = random.randrange(0, KEYRANGE, 1)
    # value = random.gauss(mu=0.0, sigma=VALRANGE)
    value = random.randint(0, VALRANGE)

    f.write(f'{key} {int(value)}\n')

f.close()