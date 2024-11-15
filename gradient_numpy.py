import numpy as np
import torch

# X = np.array([1, 2, 3, 4], dtype=np.float32)
# Y = np.array([2, 4, 6, 8], dtype=np.float32)
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

# w = 0.0 #scalar
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

#predict
def forward(X):
    return w * X

#loss = MSE
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

#gradient
#MSE = 1/N * (w*x - y)**2
#dJ/dw = 1/N 2x(w*x - y)
def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted - y)/len(x)

print("X: ")
print(X)
print(f'(before) Predict f(X): {forward(X)}')

#training
n_iters = 10
learning_rate = 0.1
for epoch in range(n_iters):
    # predict
    y_pred = forward(X)
    
    # loss
    l = loss(Y, y_pred)

    # grad
    # dw = gradient(X, Y, y_pred)
    l.backward()
    
    # gradient descent
    # w -= learning_rate * dw
    with torch.no_grad():
        w -= learning_rate * w.grad
    w.grad.zero_()
    
    # print
    if (epoch%1 == 0):
        print(f'epoch {epoch+1}: w = {w:.3f}, l = {l:.8f}')

print(f'(after) Predict f(X): {forward(X)}')
print(f'w: {w}')
print(f'loss: {loss(Y, forward(X))}')