import numpy as np

from NeuralNetworks.network import Network

def doProb2():
    # Build Network
    net = Network(3, 3)
    net.layers[0].weights = np.array([[-1, 1], [-2, 2], [-3, 3]])
    net.layers[1].weights = np.array([[-1, 1], [-2, 2], [-3, 3]])
    net.layers[2].weights = np.array([-1, 2, -1.5])

    print("Forward Pass\n")
    y = net.forward_pass(np.array([1, 1, 1]))
    print(y)

    print("Gradients\n")
    net.backpropagate(1, [1, 1, 1])

if __name__ == '__main__':
    print("Homework 5 - Begin\n\n")
    doProb2()