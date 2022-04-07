import numpy as np
import matplotlib.pyplot as plt


def init_weights(n_input, n_layer, n_output, seed=10):
    np.random.seed(seed)
    w1 = np.random.rand(n_input, n_layer)    # shape is (2, 8)
    w2 = np.random.rand(n_layer, n_output)   # shape is (8, 1)
    return w1, w2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward(x, w):
    h = w.T @ x
    h = sigmoid(h)

    return h


def EI_ft(w, h1, h2, EI2=0):
    if EI2 == 0:                     # Define different EI for the first EI
        EI = (h1 - h2) * h1 * (1 - h1)
    else:
        EI = (w @ EI2.T) * h1 * (1 - h1)

    return EI


def backward(w, h, n1, n2, EI, lr=0.01):
    h_hat = h.reshape(n1, 1)
    EI_hat = EI.reshape(1, n2)
    w -= lr * h_hat @ EI_hat

    return w


def compute_loss(y, t):
    E = 1/2 * (y - t) ** 2

    return E


# Initialization
epochs = 101
input_dim = 2
hidden_dim = 8
output_dim = 1

with open('training.txt', 'r') as f:  # open file
    f = f.readlines()  # read lines
    train_data = [[float(line.strip().split()[0]), float(line.strip().split()[1])] for line in f]
    label = [float(line.strip().split()[2]) for line in f]

x, t = np.array(train_data), np.array(label)
w1, w2 = init_weights(input_dim, hidden_dim, output_dim)
loss = []  # Record each train_loss

for epoch in range(epochs):

    loss_ = 0

    for x_i, t_i in zip(x, t):

        # forward
        # Input to hidden
        h_i = forward(x_i, w1)
        # Hidden to output
        y_i = forward(h_i, w2)

        loss_ += compute_loss(y_i, t_i) # Accumulate train_loss

        # backward(updates)
        EI2 = EI_ft(0, y_i, t_i)
        EI1 = EI_ft(w2, h_i, y_i, EI2)
        w2 = backward(w2, h_i, hidden_dim, output_dim, EI2)
        w1 = backward(w1, x_i, input_dim, hidden_dim, EI1)

    loss.append(loss_ / len(train_data))

    if epoch % 10 == 0:
        print('Epoch: %d, Loss: %f' % (epoch, loss_ / len(train_data)))


plt.plot(loss, linewidth=3)
plt.title('Loss VS Epochs')
plt.xlabel('Epochs')
plt.savefig('train_loss.png')
plt.show()
