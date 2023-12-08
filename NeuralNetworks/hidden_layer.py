import numpy as np
import math

class HiddenLayer:
    def __init__(self, num_nodes, num_nodes_prev, weights, zeroInit=False):
        self.nodes = np.zeros(num_nodes)
        if zeroInit:
            self.weights = np.zeros((num_nodes_prev, num_nodes-1))
        elif weights == []:
            self.weights = np.random.randn(num_nodes_prev, num_nodes-1)
        else:
            self.weights = weights

    def sigmoid(self, x):
        return 1 / (1+math.exp(-x))

    def forward_pass(self, prev_layer_nodes):
        self.nodes[0] = 1
        for i in range(np.size(self.nodes,0)-1):
            self.nodes[i+1] = self.sigmoid(np.dot(prev_layer_nodes, self.weights[:, i]))

        return self.nodes

    def calc_grads(self, next_layer_node_grads, next_layer_nodes, next_layer_weights, prev_layer_nodes, next_layer_output, printOut=False):
        node_grads = np.zeros(np.shape(self.nodes))
        for i in range(np.size(self.nodes, 0)):
            # New_partials - partials for every node in next layer with respect to current node
            if next_layer_output:
                new_partials = [next_layer_weights[i]]
                node_grads[i] = np.sum(np.dot(next_layer_node_grads, new_partials))
            else:
                new_partials = [next_layer_weights[i,j]*next_layer_nodes[j+1]*(1-next_layer_nodes[j+1]) for j in range(np.size(next_layer_nodes)-1)]
                node_grads[i] = np.sum(np.dot(next_layer_node_grads[1:], new_partials))

        weight_grads = np.zeros(np.shape(self.weights))
        for i in range(np.size(self.weights, 0)):
            for j in range(np.size(self.weights, 1)):
                weight_grads[i,j] = node_grads[j+1] * self.nodes[j+1] * (1 - self.nodes[j+1]) * prev_layer_nodes[i] #Offset nodes by 1 bc first node is bias

        self.node_grads = node_grads
        self.weight_grads = weight_grads

        if printOut:
            print("Hidden Layer Node Grads:")
            print(node_grads)
            print("Hidden Layer Weight Grads:")
            print(weight_grads)

        return (node_grads, weight_grads)

    def update(self, r):
        for i in range(np.size(self.weights, 0)):
            for j in range(np.size(self.weights, 1)):
                self.weights[i,j] = self.weights[i,j] - self.weight_grads[i,j]*r

