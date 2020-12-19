from __future__ import print_function

import numpy as np
import torch

filename = "../model/rnn_1_3"
model_name = filename.split(sep="/")[-1]
print(model_name)

lr_values = np.logspace(-4, 0, num=5)  # from 1e-4 to 1
print("lr_values", lr_values)
hl_values = np.array([50, 100, 300, 400, 500])
print(hl_values)

lst = [2, 3, 4]
a, b, c = lst
print(a, b, c)

x = np.random.random((5, 3))
print(type(x))
print(x.shape)
x = np.reshape(x, (5, 1, 3))
print(x.shape)
x_in = torch.tensor(x).float()
print("x_in shape", x_in.shape)
print(x_in)

print(x_in.size(), x_in.size(0), x_in.size(2))
x_flatten = x_in.view(x_in.size(0), -1)
print(x_flatten.size())
x_flatten_2 = x_in.view(1, -1)
print(x_flatten_2.size(),  x_flatten_2.shape)

y1 = torch.tensor(np.array([2, 3, 4]), dtype=torch.float)
y2 = torch.tensor(np.array([1, 2, 3]), dtype=torch.float)
loss_fun = torch.nn.MSELoss(reduction='mean')
loss = loss_fun(y1, y2)
print("Loss is ", loss)

x1 = np.array([[1, 2], [2, 3]])
x2 = np.array([[3, 4], [4, 3]])
diff = x1 - x2
print(diff)
print(np.square(diff))
print(np.mean(np.square(diff)))
