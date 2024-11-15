#!/bin/python3
# 1. Design model (input, output size, model p_testass)
# 2. Construct loss, optimizer
# 3. Training loop:
#    - model p_testass: Predict
#    - Backward pass: Gradients
#    - Update weights

import numpy as np
import torch
import torch.nn as nn

n_iters = 100
learning_rate = 0.01

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features

#design pytorch model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)
    
# model = nn.Linear(input_size, output_size)
model = LinearRegression(input_size, output_size)

#loss = MSE
loss = nn.MSELoss()

#gradient
#MSE = 1/N * (w*x - y)**2
#dJ/dw = 1/N 2x(w*x - y)
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

print("X: ")
print(X)
print(f'(before) Predict f(X): {model(X_test)}')

#training
for epoch in range(n_iters):
    # predict
    y_pred = model(X)
    
    # loss
    l = loss(Y, y_pred)

    # grad
    l.backward()
    
    # gradient descent
    optimizer.step()

    optimizer.zero_grad()
        
    # print
    if ((epoch+1)%10 == 0):
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, l = {l:.8f}')

print(f'(after) Predict f(X): {model(X_test)}')
print(f'w: {w}')
print(f'loss: {loss(Y, model(X_test))}')