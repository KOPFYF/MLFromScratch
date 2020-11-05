import torch
import torch.nn as nn
import numpy as np
print(torch.__version__)

X = torch.tensor(([2, 9], [1, 5], [3, 6]), dtype=torch.float) # 3 X 2 tensor
y = torch.tensor(([92], [100], [89]), dtype=torch.float) # 3 X 1 tensor
xPredicted = torch.tensor(([4, 8]), dtype=torch.float) # 1 X 2 tensor

print(X.size())
print(y.size())

# scale units
X_max, _ = torch.max(X, 0)
xPredicted_max, _ = torch.max(xPredicted, 0)

X = torch.div(X, X_max)
xPredicted = torch.div(xPredicted, xPredicted_max)
y = y / 100  # max test score is 100

# setting the random seed for pytorch and initializing two tensors
torch.manual_seed(42)
a = torch.randn(3,3)
b = torch.randn(3,3)

# matrix addition
print(torch.add(a,b), '\n')

# matrix subtraction
print(torch.sub(a,b), '\n')

# matrix multiplication
print(torch.mm(a,b), '\n')

# matrix division
print(torch.div(a,b), '\n')

# concatenating vertically
print(torch.cat((a,b)))

a = torch.randn(2,4)
print(a)
# reshaping tensor
b = a.reshape(1,8)
print(b)

# initializing a numpy array
a = np.array([[1,2],[3,4]])
print(a, '\n')

# converting the numpy array to tensor
tensor = torch.from_numpy(a)
print(tensor)


# initializing a tensor
a = torch.ones((2,2), requires_grad=True)
# performing operations on the tensor
b = a + 5
c = b.mean()
print(b,c)
# back propagating
c.backward()
# computing gradients
print(a.grad)


