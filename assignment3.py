import numpy as np
import matplotlib.pyplot as plt


class Neural_Network():
    def __init__(self, input_nodes_num, hidden_nodes_num, output_nodes_num, lr):
        # Initialization of the size of neurons
        self.input_nodes = input_nodes_num
        self.hidden_nodes = hidden_nodes_num
        self.output_nodes = output_nodes_num
        self.learning_rate = lr

        # Initialization weights (randomly, mean is 0, dev sqrt(neurons))
        self.w_input_hidden = np.random.normal(0.0, pow(self.hidden_nodes, -0.5),
                                                  (self.hidden_nodes, self.input_nodes))
        self.w_hidden_output = np.random.normal(0.0, pow(self.output_nodes, -0.5),
                                                   (self.output_nodes, self.hidden_nodes))

        # lambda defines activation function (sigmoid)
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))
        pass


    def forward(self, inputs_list):
        # row vector to column vector (for dot operation)
        inputs = np.array(inputs_list, ndmin=2).T
        # Weighted sum and sigmoid function
        hidden_inputs = np.dot(self.w_input_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        # Weighted sum and sigmoid function (to get final output)
        final_inputs = np.dot(self.w_hidden_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs.squeeze()


    def compute_loss(self, inputs_list, targets_list):
        final_outputs = self.forward(inputs_list)
        loss = 1 / 2 * (final_outputs - targets_list) ** 2
        loss = sum(loss) / len(inputs_list)

        return loss


    def train(self, inputs_list, targets_list, optimizer='gradient_descent'):
        # row vector to column vector
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2)

        hidden_inputs = np.dot(self.w_input_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.w_hidden_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs).squeeze()

        # loss function
        output_errors = targets - final_outputs

        hidden_errors = np.dot(self.w_hidden_output.T, output_errors)
        # Update weights
        gradient_o_h = np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        gradient_h_i = np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        if optimizer == 'gradient_descent':
            self.w_hidden_output += self.learning_rate * gradient_o_h
            self.w_input_hidden += self.learning_rate * gradient_h_i

        elif optimizer == 'Adagrad':
            eps = 1e-5
            square_gradient = gradient_o_h ** 2
            self.w_hidden_output += self.learning_rate * (1 / np.sqrt(square_gradient + eps)) * gradient_o_h
            square_gradient = gradient_h_i ** 2
            self.w_input_hidden += self.learning_rate * (1 / np.sqrt(square_gradient + eps)) * gradient_h_i


input_nodes = 2
hidden_nodes = 8
output_nodes = 1
learning_rate = 0.0001

Network = Neural_Network(input_nodes, hidden_nodes, output_nodes, learning_rate)


with open('training.txt', 'r') as f:  # open file
    f = f.readlines()  # read lines

    train_data, train_label = [], []
    for line in f:
        line = line.strip().split()
        train_data.append([float(line[0]), float(line[1])])
        train_label.append(float(line[2]))

with open('test.txt', 'r') as f:
    f = f.readlines()  # read lines

    test_data, test_label = [], []
    for line in f:
        line = line.strip().split()
        test_data.append([float(line[0]), float(line[1])])
        test_label.append(float(line[2]))

train_x, train_t = np.array(train_data), np.array(train_label)
test_x, test_t = np.array(test_data), np.array(test_label)
train_loss_history, test_loss_history = [], []  # record loss after each epoch

# Question a
for epoch in range(200):
    #Network.train(train_x, train_t)
    state = np.random.get_state()
    np.random.shuffle(train_x)
    np.random.set_state(state)
    np.random.shuffle(train_t)
    for x_i, t_i in zip(train_x, train_t):
        inputs = x_i
        targets = t_i

        Network.train(inputs, targets)


    train_loss = Network.compute_loss(train_x, train_t)
    test_loss = Network.compute_loss(test_x, test_t)
    train_loss_history.append(train_loss)
    test_loss_history.append(test_loss)
    print('Epoch %d, train_loss: %f, validation_loss: %f' % (epoch, train_loss, test_loss))

print('train successful!')

plt.plot(train_loss_history, label='Training loss', linewidth=3)
plt.plot(test_loss_history, label='Testing loss', linewidth=3)
plt.title('Loss VS Epochs')
plt.xlabel('Epochs')
plt.legend(loc='center right')
plt.savefig('train_test_loss.png')
plt.show()
plt.close()


# Question b
Network = Neural_Network(input_nodes, hidden_nodes, output_nodes, learning_rate)
loss_history = []
for num_hidden_nodes in range(2, 17):
    Network = Neural_Network(input_nodes, num_hidden_nodes, output_nodes, learning_rate)
    for epoch in range(200):
        state = np.random.get_state()
        np.random.shuffle(train_x)
        np.random.set_state(state)
        np.random.shuffle(train_t)
        for x_i, t_i in zip(train_x, train_t):
            inputs = x_i
            targets = t_i

            Network.train(inputs, targets)


    loss = Network.compute_loss(train_x, train_t)
    loss_history.append(loss)
    print('Hidden_nodes: %d, loss: %f' % (num_hidden_nodes, loss))


x_coordinates = list(range(2, 17))
plt.plot(x_coordinates, loss_history, linewidth=3)
plt.title('Loss VS hidden layer size')
plt.xlabel('Hidden_layer_size')
plt.savefig('different_hidden_nodes_loss.png')
plt.show()
plt.close()


# Question c
learning_rate = 0.01
Network = Neural_Network(input_nodes, hidden_nodes, output_nodes, learning_rate)
train_loss_history, test_loss_history = [], []
for epoch in range(200):
    #Network.train(train_x, train_t)
    state = np.random.get_state()
    np.random.shuffle(train_x)
    np.random.set_state(state)
    np.random.shuffle(train_t)
    for x_i, t_i in zip(train_x, train_t):
        inputs = x_i
        targets = t_i

        Network.train(inputs, targets, optimizer='Adagrad')


    train_loss = Network.compute_loss(train_x, train_t)
    test_loss = Network.compute_loss(test_x, test_t)
    train_loss_history.append(train_loss)
    test_loss_history.append(test_loss)
    print('Epoch %d, train_loss: %f, validation_loss: %f' % (epoch, train_loss, test_loss))


plt.plot(train_loss_history, linewidth=3)
plt.title('Loss VS Epochs')
plt.xlabel('Epochs')
plt.savefig('train_loss_adagrad.png')
plt.show()
plt.close()
