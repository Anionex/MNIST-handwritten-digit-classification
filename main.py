import numpy as np
hidden_layer_neurons = 24
def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1. - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def initialize_parameters():
    weights = [
        np.random.randn(img_size * img_size, hidden_layer_neurons) * np.sqrt(2. / (img_size * img_size)),
        np.random.randn(hidden_layer_neurons, hidden_layer_neurons) * np.sqrt(2. / hidden_layer_neurons),
        np.random.randn(hidden_layer_neurons, 10) * np.sqrt(2. / hidden_layer_neurons)
    ]
    biases = [np.zeros((1, hidden_layer_neurons)), np.zeros((1, hidden_layer_neurons)), np.zeros((1, 10))]
    return weights, biases


def forward_propagation(X, weights, biases):
    z1 = np.dot(X, weights[0]) + biases[0]
    a1 = relu(z1)
    z2 = np.dot(a1, weights[1]) + biases[1]
    a2 = relu(z2)
    z3 = np.dot(a2, weights[2]) + biases[2]
    a3 = softmax(z3)

    return a1, a2, a3, z1, z2, z3
def compute_loss(a3, y):
    m = y.shape[0]
    return np.sum((y - a3) ** 2) / m

def backward_propagation(X, a1, a2, a3, y, weights, z1, z2, z3): 
    # y是one-hot标签
    m = y.shape[0]
    # a3_delta = (y - a3) * d_relu(z3)
    a3_delta = a3 - y  #省略常数2， 因为不影响梯度的方向

    a2_delta = np.dot(a3_delta, weights[2].T) * d_relu(z2)

    a1_delta = np.dot(a2_delta, weights[1].T) * d_relu(z1)

    grads_w = [
        np.dot(X.T, a1_delta) / m,
        np.dot(a1.T, a2_delta) / m,
        np.dot(a2.T, a3_delta) / m
    ]

    grads_b = [
        np.sum(a1_delta, axis=0, keepdims=True) / m,
        np.sum(a2_delta, axis=0, keepdims=True) / m,
        np.sum(a3_delta, axis=0, keepdims=True) / m
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

def compute_accuracy(weights, biases, img_size=28, batch_size=100):
    data = np.load('mnist.npz')
    X_test = data['x_test'][:batch_size]
    y_test = data['y_test'][:batch_size]

    X_test = X_test.reshape(batch_size, img_size * img_size) / 255.0

    y = np.zeros((batch_size, 10))
    y[np.arange(batch_size), y_test] = 1

    _, _, a3, _, _, _ = forward_propagation(X_test, weights, biases)

    accuracy = np.sum(np.argmax(a3, axis=1) == np.argmax(y, axis=1)) / batch_size
    return accuracy

def save_model(weights, biases):
    np.savez('model.npz', w0=weights[0], w1=weights[1], w2=weights[2], b0=biases[0], b1=biases[1], b2=biases[2])


import matplotlib.pyplot as plt


def plot_training(history):
    loss, accuracy = history['loss'], history['accuracy']

    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, 'ro-', label='Training accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    batch_size = 100
    epochs = 100
    total = 10000
    # define hyperparams
    learning_rate = 0.01
    tolerance = 0.0001
    threshold = 0.94
    losses = []
    accuracies = []
    accuracy = 0.
    # 初始化参数，用了一种方法叫Xavier/zeɪviər/初始化，适用于激活函数为 sigmoid 或 tanh 的神经网络层。
    weights, biases = initialize_parameters()
    for epoch in range(epochs):
        index = 0
        epoch_losses = []
        epoch_accuracies = []
        # learning_rate /= 2
        while index < total:
            if index + batch_size > total:
                index = 0
            X, y = load_data(index, batch_size)
            # z是线性组合（加权和和bias的和）
            a1, a2, a3, z1, z2, z3 = forward_propagation(X, weights, biases)
            loss = compute_loss(a3, y)

            accuracy = compute_accuracy(weights, biases)

            epoch_losses.append(loss)
            epoch_accuracies.append(accuracy)

            if index % 1000 == 0:
                print(f"Epoch {epoch + 1}, Batch {index // batch_size}, Loss: {loss:.4f}, Accuracy: {accuracy:.4%}")
            grads_w, grads_b = backward_propagation(X, a1, a2, a3, y, weights, z1, z2, z3)
            weights, biases = update_parameters(learning_rate, grads_w, grads_b, weights, biases)

            index += batch_size
        avg_accuracy = np.mean(epoch_accuracies)
        avg_losses = np.mean(epoch_losses)
        losses.append(avg_losses)
        accuracies.append(avg_accuracy)


        if avg_accuracy > threshold:
            print(f"Done!:accuracy:{accuracy}\nepochs:{epoch}")
            save_model(weights, biases)
            break

    print("Training finished")
    print("Test.")

    accuracy = compute_accuracy(weights, biases, batch_size=10000)
    print(f"General accuracy: {accuracy:.4%}\n")

    history = {'loss': losses, 'accuracy': accuracies}
    plot_training(history)

if __name__ == '__main__':
    main()
