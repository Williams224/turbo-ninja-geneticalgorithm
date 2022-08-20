import numpy as np


def fast_random_choice(l):
    return l[int(np.random.random() * len(l))]
