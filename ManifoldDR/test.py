import numpy as np

def softmax(x, T):
    sum = 0
    for i in x:
        sum += np.exp(i/T)

    for i in x:
        print(np.exp(i/T)/sum)


softmax([0.8, 3, 0.01], T=10)