import random

from pandas import array


def get_random_array(l, int_range=None):
    arr = []
    for i in range(l):
        if int_range is None:
            arr.append(random.random())
        else:
            a, b = int_range
            arr.append(random.randint(a, b))
    return arr
