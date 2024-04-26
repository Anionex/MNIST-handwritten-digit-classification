import numpy as np

data = np.load('model.npz')
print(data.files)
print(data['w0'].shape)
print(data['w1'])
print(data['w1'].shape)

weights = {
    'w0': data['w0'],
    'w1': data['w1'],
    'w2': data['w2'],
}
biases = {
    'b0': data['b0'],
    'b1': data['b1'],
    'b2': data['b2'],
}



