import numpy as np

def sample(percent,buy_rate):
    positive, buy = False, False
    sample1 = np.random.rand()
    if sample1 < percent:
        positive = True
        sample2 = np.random.rand()
        if sample2 < buy_rate:
            buy = True
    return positive, buy