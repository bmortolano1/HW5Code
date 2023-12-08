import numpy as np

class OutputLayer:
    def __init__(self, num_nodes_prev, weights, zeroInit=False):
        self.nodes = np.zeros(1)
        if zeroInit:
            self.weights = np.zeros(num_nodes_prev)
        elif weights == []:
            self.weights = np.random.randn(num_nodes_prev)
        else:
            self.weights = weights

    def forward_pass(self, prev_layer_nodes):
        self.nodes[0] = 1
        y = np.dot(prev_layer_nodes, self.weights)

        return y

    def calc_grads(self, y, y_star, prev_layer_nodes, printOut=False):
        out_grad = y - y_star

        weight_grads = np.zeros(np.size(self.weights))
        for i in range(np.size(self.weights, 0)):
            weight_grads[i] = out_grad * prev_layer_nodes[i]

        self.node_grads = out_grad
        self.weight_grads = weight_grads

        if printOut:
            print("Output Layer Node Grads:")
            print(self.node_grads)
            print("Output Layer Weight Grads:")
            print(weight_grads)

        return weight_grads

    def update(self, r):
        for i in range(np.size(self.weights)):
            self.weights[i] = self.weights[i] - self.weight_grads[i]*r


