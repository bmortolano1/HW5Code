import numpy as np
from NeuralNetworks.hidden_layer import HiddenLayer
from NeuralNetworks.output_layer import OutputLayer

class Network:
    def __init__(self, in_features, hidden_nodes, num_layers, zeroInit=False):

        self.layers = np.empty(0)

        # First Layer
        self.layers = np.append(self.layers, HiddenLayer(hidden_nodes, in_features, [], zeroInit=zeroInit))

        # Hidden layers
        for i in range(1, num_layers-1):
            self.layers = np.append(self.layers, HiddenLayer(hidden_nodes, hidden_nodes, [], zeroInit=zeroInit))

        # Output layer
        self.layers = np.append(self.layers, OutputLayer(hidden_nodes, [], zeroInit=zeroInit))

    def forward_pass(self, input, printOut=False):
        nodes = input
        for layer in self.layers[0:-1]:
            nodes = layer.forward_pass(nodes)
            if printOut:
                print(nodes)

        y = self.layers[-1].forward_pass(nodes)
        return y

    def backpropagate(self, y_star, input, printOut=False):
        y = self.forward_pass(input)
        self.layers[-1].calc_grads(y, y_star, self.layers[-2].nodes, printOut)
        for i in list(reversed(range(1, np.size(self.layers)-1))):
            (node_grads, weight_grads) = self.layers[i].calc_grads(self.layers[i + 1].node_grads,
                                                                   self.layers[i + 1].nodes, self.layers[i + 1].weights,
                                                                   self.layers[i - 1].nodes,
                                                                   (i >= np.size(self.layers) - 2), printOut)

        (node_grads, weight_grads) = self.layers[0].calc_grads(self.layers[1].node_grads, self.layers[1].nodes,
                                                               self.layers[1].weights, input,
                                                               False, printOut)

    def update_weights(self, r):
        for layer in self.layers:
            layer.update(r)

    def train(self, features, labels, epochs, gamma0, d):
        rng = np.random.default_rng()
        indeces = np.arange(np.size(labels))
        n = 0

        epoch_accuracies = []

        for T in range(epochs):
            rng.shuffle(indeces)
            n_corr = 0
            n_incorr = 0

            for i in indeces:
                x = features[i]
                x = np.append(1, x)
                y_star = labels[i]
                gamma = gamma0 / (1 + gamma0/d*T) # Decay gamma

                y = self.forward_pass(x)

                if (y * y_star) > 0:
                    n_corr = n_corr + 1
                else:
                    n_incorr = n_incorr + 1

                self.backpropagate(y_star, x)
                self.update_weights(gamma)

                n = n + 1

            epoch_accuracies.append(n_corr / (n_corr + n_incorr))
            if T > 10:
                if epoch_accuracies[-1] <= np.average(epoch_accuracies[-10:])*1.02:
                    break

        return epoch_accuracies[-1]

    def test(self, features, labels):
        n_corr = 0
        n_incorr = 0

        for i in np.arange(np.size(labels)):
            x = features[i]
            x = np.append(1, x)
            y_star = labels[i]

            y = self.forward_pass(x)

            if (y * y_star) > 0:
                n_corr = n_corr + 1
            else:
                n_incorr = n_incorr + 1

        return (n_corr / (n_corr + n_incorr))