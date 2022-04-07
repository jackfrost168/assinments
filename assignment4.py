import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE


G = nx.Graph()  # Graph initialization
with open('assignment4/Assignment4.edgelist', 'r') as f:  # open file
    f = f.readlines()  # read lines
    for edge in f:
        node1, node2, _ = edge.split()
        G.add_edge(int(node1), int(node2))

eta = 0.0005 # learning rate

A = np.array(nx.adjacency_matrix(G).todense())  # Adjacency matrix
Z = np.random.rand(G.number_of_nodes(), 4)      # Embeddings

loss_history = []

# SGD update the embedding
for epoch in range(2000):
    for u in range(G.number_of_nodes()):
        for v in range(u, G.number_of_nodes()):
            e_u_v = np.dot(Z[u], Z[v]) - A[u][v]
            Z[u], Z[v] = Z[u] - eta * e_u_v * Z[v], Z[v] - eta * e_u_v * Z[u]
    loss = np.sum(np.power(np.dot(Z, Z.T) - A, 2))
    loss_history.append(loss)
    if epoch % 100 == 0:
        print('Epoch:', epoch, 'loss:', loss)

plt.plot(loss_history, label='Training loss', linewidth=3)
#plt.title('Loss VS Epochs')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.savefig('loss.png')
plt.show()
plt.close()

# Question b
with open('assignment4/karate_label.txt', 'r') as f:
    f = f.readlines()
    labels = []
    for line in f:
        node, label = line.split()
        labels.append([int(node), int(label)])

tsne = TSNE(n_components=2, perplexity=16.0)  # Using t-SNE as the visualization tool
Y = tsne.fit_transform(Z)
for i, node in enumerate(G.nodes):
    if labels[node][1] == 0:
        plt.scatter(Y[i][0], Y[i][1], c='g')
    else:
        plt.scatter(Y[i][0], Y[i][1], c='r')

#plt.title('t-SNE visualization')
plt.savefig('t-SNE.png')
plt.show()
plt.close()