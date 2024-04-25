import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def initialize_parameters():
    weights = [
        np.random.randn(img_size * img_size, 16) * np.sqrt(1. / (img_size * img_size)),
        np.random.randn(16, 16) * np.sqrt(1. / 16),
        np.random.randn(16, 10) * np.sqrt(1. / 16)
    ]
    biases = [np.zeros((1, 16)), np.zeros((1, 16)), np.zeros((1, 10))]
    return weights, biases


def forward_propagation(X, weights, biases):
    z = [np.zeros((100, 16)), np.zeros((100, 16)), np.zeros((100, 10))]
    z[0] = np.dot(X, weights[0]) + biases[0]
    layer1 = sigmoid(z[0])
    z[1] = np.dot(layer1, weights[1]) + biases[1]
    layer2 = sigmoid(z[1])
    z[2] = np.dot(layer2, weights[2]) + biases[2]
    layer3 = sigmoid(z[2])

    return layer1, layer2, layer3, z
def compute_loss(layer3, y):
    m = y.shape[0]
    return np.sum((y - layer3) ** 2) / m

def backward_propagation(X, layer1, layer2, layer3, y, weights, z):
    layer3_error = y - layer3
    layer3_delta = layer3_error * d_sigmoid(z[2])

    layer2_error = np.dot(layer3_delta, weights[2].T)
    layer2_delta = layer2_error * d_sigmoid(z[1])

    layer1_error = np.dot(layer2_delta, weights[1].T)
    layer1_delta = layer1_error * d_sigmoid(z[0])

    grads_w = [
        np.dot(X.T, layer1_delta),  # Correct: transpose X, not layer1_delta
        np.dot(layer1.T, layer2_delta),  # Correct: transpose layer1, not layer2_delta
        np.dot(layer2.T, layer3_delta)  # Correct: transpose layer2, not layer3_delta
    ]

    grads_b = [
        np.sum(layer1_delta, axis=0, keepdims=True),
        np.sum(layer2_delta, axis=0, keepdims=True),
        np.sum(layer3_delta, axis=0, keepdims=True)
    ]

    return grads_w, grads_b


def update_parameters(learning_rate, grads_w,grads_b, weights, biases):
    for i in range(len(weights)):
        weights[i] -= learning_rate * grads_w[i]
        biases[i] -= learning_rate * grads_b[i]
    return weights, biases

def load_data(index, batch_size):
    try:
        X = np.zeros((batch_size, img_size * img_size))
        y = np.zeros((batch_size, 10))
        data = np.load('mnist.npz')

        X = data['x_train'][index:index + batch_size]
        X = X.reshape(batch_size, img_size * img_size)

        X = X / 255.0
        y[np.arange(batch_size), data['y_train'][index:index + batch_size]] = 1
    except IOError as e:
        print(f"Error loading data: {e}")
        return None, None
    return X, y


img_size = 28


import numpy as np

def compute_accuracy(weights, biases, img_size=28, batch_size=100):
    data = np.load('mnist.npz')
    X_test = data['x_test'][:batch_size]  # Load only batch_size number of images
    y_test = data['y_test'][:batch_size]  # Load corresponding labels

    # Reshape and normalize the test images
    X_test = X_test.reshape(batch_size, img_size * img_size) / 255.0

    # Convert labels to one-hot vectors
    y = np.zeros((batch_size, 10))
    y[np.arange(batch_size), y_test] = 1

    # Perform forward propagation on the test data
    _, _, layer3, _ = forward_propagation(X_test, weights, biases)

    # Compute accuracy
    accuracy = np.sum(np.argmax(layer3, axis=1) == np.argmax(y, axis=1)) / batch_size
    return accuracy






def main():
    batch_size = 100
    epoch = 100

    learning_rate = 0.001
    tolerance = 0.0001
    losses = []
    accuracies = []
    index = 0

    weights, biases = initialize_parameters()
    for i in range(epoch):
        print("Epoch: ", i)

        index += batch_size
        X, y = load_data(index, batch_size)

        layer1, layer2, layer3, z = forward_propagation(X, weights, biases)
        loss = compute_loss(layer3, y)

        accuracy = compute_accuracy(weights, biases)
        accuracies.append(accuracy)
        print("Accuracy: ", accuracy)

        # if i > 0 and abs(accuracies[i] - accuracies[i - 1] ) < tolerance:
        #     print("Accuracy converged")
        #     break


        losses.append(loss)
        print( "Loss: ", loss)
        grads_w, grads_b = backward_propagation(X, layer1, layer2, layer3, y, weights, z)
        # print("Grads_w: ", grads_w)
        # print("Grads_b: ", grads_b)
        weights, biases = update_parameters(learning_rate, grads_w, grads_b, weights, biases)



        # if i > 0 and abs(losses[i] - losses[i - 1]) < tolerance:
        #     print("Loss converged")
        #     break

    
    print("Training finished")
    

if __name__ == '__main__':
    main()
