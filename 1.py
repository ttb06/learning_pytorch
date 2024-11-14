#! /bin/python3

import torch as t
import numpy as np

a = np.array([2, 3])
b = t.from_numpy(a, dtype=t.float64)
print (a)
print (b)
