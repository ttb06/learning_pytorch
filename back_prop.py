import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)
print(x)
w = torch.tensor(1.0, requires_grad=True)

y_hat = x * y
loss = (y_hat - w).pow(2)
print("loss: ", loss)

#backward
loss.backward()
print(w.grad)
