import numpy as np
import math

class Layer:
    def __init__(self, num_nodes, num_nodes_prev, weights):
        self.nodes = np.zeros(num_nodes)
        if weights == []:
            self.weights = np.random.rand(num_nodes_prev, num_nodes-1)
        else:
            self.weights = weights

    def sigmoid(self, x):
        return 1 / (1+math.exp(x))

    def forward_pass(self, prev_layer_nodes, activate):
        self.nodes[0] = 1
        for i in range(np.size(self.nodes,0)-1):
            if activate:
                self.nodes[i+1] = self.sigmoid(np.dot(prev_layer_nodes, self.weights[:, i]))
            else:
                self.nodes[i+1] = np.dot(prev_layer_nodes, self.weights[:, i])

        return self.nodes

    def calc_grads(self, next_layer_node_grads, next_layer_nodes, next_layer_weights, prev_layer_nodes, sigmoid, next_layer_sigmoid):
        node_grads = np.zeros(np.shape(self.nodes))
        for i in range(np.size(self.nodes, 0)):
            # New_partials - partials for every node in next layer with respect to current node
            if next_layer_sigmoid:
                new_partials = [next_layer_weights[i,j]*next_layer_nodes[j]*(1-next_layer_nodes[j]) for j in range(np.size(next_layer_nodes,1))]
            else:
                new_partials = [next_layer_weights[i, j] for j in
                                range(np.size(next_layer_nodes, 1))]
            node_grads[i] = np.sum(np.dot(next_layer_node_grads, new_partials))

        weight_grads = np.zeros(np.size(self.weights))
        for i in range(np.size(self.weights, 0)):
            for j in range(np.size(self.weights, 1)):
                if sigmoid:
                    weight_grads[i,j] = node_grads[j] * self.nodes[j] * (1 - self.nodes[j]) * prev_layer_nodes[i]
                else:
                    weight_grads[i, j] = node_grads[j] * prev_layer_nodes[i]

        return (node_grads, weight_grads)


