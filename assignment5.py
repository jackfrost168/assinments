# -*- coding:utf-8 -*-
# Author: Yu Hou
# Date: 2022-04-09
# Decription: Deepwalk
# Reference: [1] Perozzi B, Al-Rfou R, Skiena S. Deepwalk: Online learning of social representations[C]//Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2014: 701-710.(http://www.perozzi.net/publications/14_kdd_deepwalk.pdf)

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
# The constructed graph is undirected graph
G = nx.read_edgelist('assignment5/karate_club.edgelist', create_using=nx.Graph())
#print(G.adj['0'])

# walk sequence begin with start_node
def deepwalk_walk(walk_length, start_node):
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = list(G.neighbors(cur))
        if len(cur_nbrs) > 0:
            walk.append(random.choice(cur_nbrs))
        else:
            break
    return walk


# Generate walks of random walks
def generate_walks(nodes, walks_per_vertex, walk_length):
    walks = []
    for _ in range(walks_per_vertex):
        random.shuffle(nodes)
        for v in nodes:
            walks.append(deepwalk_walk(walk_length=walk_length, start_node=v))
    return walks


class skip_gram():

    def __init__(self, input_nodes_num, hidden_nodes_num, output_nodes_num, window_size, lr, epochs):

        self.input_nodes = input_nodes_num
        self.hidden_nodes = hidden_nodes_num
        self.output_nodes = output_nodes_num
        self.lr = lr
        self.epochs = epochs
        self.window_size = window_size

    def generate_training_data(self, walks):

        training_data = []

        # Cycle through each sentence in walks
        for sentence in walks:
            sent_len = len(sentence)

            # Cycle through each word in sentence
            for i, word in enumerate(sentence):
                # Convert target word to one-hot
                w_target = self.word2onehot(sentence[i])

                # Cycle through context window
                w_context = []

                for j in range(i - self.window_size // 2, i + self.window_size // 2 + 1):
                    if j != i and j <= sent_len - 1 and j >= 0:
                        # Append the one-hot representation of word to w_context
                        w_context.append(self.word2onehot(sentence[j]))

                training_data.append([w_target, w_context])

        return np.array(training_data)

    def word2onehot(self, word):
        # word_vec - initialize a blank vector
        word_vec = [0 for i in range(0, self.input_nodes)]

        # Get ID of word from word_index
        word_index = list(G.nodes()).index(word)

        # Change value from 0 to 1 according to ID of the word
        word_vec[word_index] = 1

        return word_vec

    def train(self, training_data):

        self.w1 = np.random.uniform(-1, 1, (self.input_nodes, self.hidden_nodes))
        self.w2 = np.random.uniform(-1, 1, (self.hidden_nodes, self.input_nodes))

        # Cycle through each epoch
        loss_history = []
        for i in range(self.epochs):
            # Initialize loss to 0
            self.loss = 0
            # Cycle through each training sample
            # w_t = vector for target word, w_c = vectors for context words
            for w_t, w_c in training_data:
                # Forward pass
                y_pred, h, u = self.forward_pass(w_t)

                # Calculate error
                # 1. For a target word, calculate difference between y_pred and each of the context words
                # 2. Sum up the differences using np.sum to give us the error for this particular target word
                EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)

                # Backpropagation
                # We use SGD to backpropagate errors - calculate loss on the output layer
                self.backprop(EI, h, w_t)

                # Calculate loss
                # There are 2 parts to the loss function
                # Part 1: -ve sum of all the output +
                # Part 2: length of context words * log of sum for all elements (exponential-ed) in the output layer before softmax (u)

                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))

            loss_history.append(self.loss / len(training_data))
            print('Epoch:', i, "Loss:", self.loss / len(training_data))

        return loss_history

    def forward_pass(self, x):
        h = np.dot(x, self.w1)
        u = np.dot(h, self.w2)
        y_c = self.softmax(u)
        return y_c, h, u

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def backprop(self, e, h, x):
        # Going backwards, we need to take derivative of E with respect of w2

        dl_dw2 = np.outer(h, e)
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))

        # Update weights
        self.w1 = self.w1 - (self.lr * dl_dw1)
        self.w2 = self.w2 - (self.lr * dl_dw2)


epochs = 300
input_nodes = nx.number_of_nodes(G)
hidden_nodes = 2
output_nodes = nx.number_of_nodes(G)
learning_rate = 0.001

d = 2 # dimension of each embedding vector (𝑑)
walk_length = 10 # walk length (l)
window_size = 3 # window size (𝑤)
walks_per_vertex = 5 # walks per vertex (𝛾)

nodes = list(G.nodes())
walks = generate_walks(nodes, walks_per_vertex=5, walk_length=10)

word2vec = skip_gram(input_nodes, hidden_nodes, output_nodes, window_size, learning_rate, epochs)
training_data = word2vec.generate_training_data(walks)
loss_history = word2vec.train(training_data)

plt.plot(loss_history, label='Training loss', linewidth=3)
#plt.title('Loss VS Epochs')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.savefig('loss_deepwalk.png')
plt.show()
plt.close()

node_features = []
print(nodes)
for node in nodes:
    node_features.append(word2vec.word2onehot(node))

_, _, embeddings = word2vec.forward_pass(node_features)

with open('assignment5/karate_label.txt', 'r') as f:
    f = f.readlines()
    labels = []
    for line in f:
        node, label = line.split()
        labels.append([node, int(label)])

for node, label in labels:
    i = nodes.index(node)
    if label == 0:
        plt.scatter(embeddings[i][0], embeddings[i][1], c='g')
    else:
        plt.scatter(embeddings[i][0], embeddings[i][1], c='r')

plt.savefig('deepwalk.png')
plt.show()
plt.close()
