import torch
import torch.nn as nn
import numpy as np

def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss

Y = np.array([1, 0, 0])

y_pred_good = np.array([0.7, 0.2, 0.1])
y_pred_bad = np.array([0.1, 0.2, 0.7])

l1 = cross_entropy(Y, y_pred_good)
l2 = cross_entropy(Y, y_pred_bad)
# print (l1, l2)

loss = nn.CrossEntropyLoss()
Y = torch.tensor([0])
#nsamples * nclasses -> 1*3
y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
y_pred_bad = torch.tensor([[0, 0, 0.1]])

l1 = loss(y_pred_bad, Y)
l2 = loss(y_pred_good, Y)
print (l1, l2)