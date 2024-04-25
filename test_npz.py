import numpy as np

data = np.load('mnist.npz')
print(data.files)
print(data['x_train'].shape)
print(data['y_train'].shape)

