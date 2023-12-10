import random

TESTNUM = 1000000
KEYSIZE = 100
KEYRANGE = 20000
VALRANGE = 10

f = open("../testcases/inputs/in.txt", "w")

# in the future, to let each key's value have a 
# distribution centered at 0, (so that we don't overflow)
# we can generate a random size for each key, and then generate a sequence. then shuffle before the next key value.
keys = [0] * KEYSIZE
for i in range(KEYSIZE):
    keys[i] = random.randrange(0, KEYRANGE, 1)

for i in range(TESTNUM):
    keyindex = random.randrange(0, KEYSIZE, 1)
    key = keys[keyindex]
    # value = random.gauss(mu=0.0, sigma=VALRANGE)
    value = random.randint(0, VALRANGE)

    f.write(f'{key} {int(value)}\n')

# for i in range(50):
#     f.write(f'{i} {int(i)}\n')

f.close()