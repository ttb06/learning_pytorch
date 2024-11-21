import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)  

x = np.array([2, 1, 0.1])
outputs = softmax(x)
print(outputs)

x = torch.from_numpy(x)
outputs = torch.softmax(x, dim = 0)
print(outputs)