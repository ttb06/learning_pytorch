import numpy as np

X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0 #scalar

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
    return np.dot(2*x, y_predicted - y).mean()

print("X: ")
print(X)
print(f'(before) Predict f(X): {forward(X)}')

#training
n_iters = 10
learning_rate = 0.01
for epoch in range(n_iters):
    # predict
    y_pred = forward(X)
    
    # loss
    l = loss(Y, y_pred)

    # grad
    dw = gradient(X, Y, y_pred)
    
    # gradient descent
    w -= learning_rate * dw

    # print
    if (epoch%1 == 0):
        print(f'epoch {epoch+1}: w = {w:.3f}, l = {l:.8f}')

print(f'(after) Predict f(X): {forward(X)}')
print(f'loss: {loss(Y, forward(X))}')