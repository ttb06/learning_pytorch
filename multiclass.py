import torch
import numpy as np
import torch.nn as nn

#softmax
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(x)
        out = self.linear2(x)
        #no softmax
        return out
    
model = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss()

