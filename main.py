import numpy as np

from NeuralNetworks.network import Network
import data_reader as dt

def doProb2():
    print("PROBLEM 2\n")
    # Build Network
    net = Network(3, 3, 3)
    net.layers[0].weights = np.array([[-1, 1], [-2, 2], [-3, 3]])
    net.layers[1].weights = np.array([[-1, 1], [-2, 2], [-3, 3]])
    net.layers[2].weights = np.array([-1, 2, -1.5])

    print("Forward Pass")
    y = net.forward_pass(np.array([1, 1, 1]), True)
    print(y)

    print("\nBackpropagation")
    net.backpropagate(1, [1, 1, 1], True)

def doProb3():
    print("PROBLEM 2b\n")

    [train_features, train_labels] = dt.parse_file("./bank-note/train.csv")
    [test_features, test_labels] = dt.parse_file("./bank-note/test.csv")

    net = Network(np.size(train_features, 1)+1,5,3)
    print("Width 5 Train Accuracy: " + str(net.train(train_features, train_labels, 100, .1, 0.05)))
    print("Width 5 Test Accuracy: " + str(net.test(test_features, test_labels)) + str("\n"))

    net = Network(np.size(train_features, 1)+1,10,3)
    print("Width 10 Train Accuracy: " + str(net.train(train_features, train_labels, 100, .1, 0.05)))
    print("Width 10 Test Accuracy: " + str(net.test(test_features, test_labels)) + str("\n"))

    net = Network(np.size(train_features, 1)+1,25,3)
    print("Width 25 Train Accuracy: " + str(net.train(train_features, train_labels, 100, .1, 0.05)))
    print("Width 25 Test Accuracy: " + str(net.test(test_features, test_labels)) + str("\n"))

    net = Network(np.size(train_features, 1)+1,50,3)
    print("Width 50 Train Accuracy: " + str(net.train(train_features, train_labels, 100, .1, 0.05)))
    print("Width 50 Test Accuracy: " + str(net.test(test_features, test_labels)) + str("\n"))

    net = Network(np.size(train_features, 1)+1,60,3)
    print("Width 100 Train Accuracy: " + str(net.train(train_features, train_labels, 100, .1, 0.05)))
    print("Width 100 Test Accuracy: " + str(net.test(test_features, test_labels)) + str("\n"))

def doProb4():
    print("PROBLEM 2c\n")

    [train_features, train_labels] = dt.parse_file("./bank-note/train.csv")
    [test_features, test_labels] = dt.parse_file("./bank-note/test.csv")

    net = Network(np.size(train_features, 1)+1,5,3, zeroInit=True)
    print("Width 5 Train Accuracy: " + str(net.train(train_features, train_labels, 100, .1, 0.05)))
    print("Width 5 Test Accuracy: " + str(net.test(test_features, test_labels)) + str("\n"))

    net = Network(np.size(train_features, 1)+1,10,3, zeroInit=True)
    print("Width 10 Train Accuracy: " + str(net.train(train_features, train_labels, 100, .1, 0.05)))
    print("Width 10 Test Accuracy: " + str(net.test(test_features, test_labels)) + str("\n"))

    net = Network(np.size(train_features, 1)+1,25,3, zeroInit=True)
    print("Width 25 Train Accuracy: " + str(net.train(train_features, train_labels, 100, .1, 0.05)))
    print("Width 25 Test Accuracy: " + str(net.test(test_features, test_labels)) + str("\n"))

    net = Network(np.size(train_features, 1)+1,50,3, zeroInit=True)
    print("Width 50 Train Accuracy: " + str(net.train(train_features, train_labels, 100, .1, 0.05)))
    print("Width 50 Test Accuracy: " + str(net.test(test_features, test_labels)) + str("\n"))

    net = Network(np.size(train_features, 1)+1,60,3, zeroInit=True)
    print("Width 100 Train Accuracy: " + str(net.train(train_features, train_labels, 100, .1, 0.05)))
    print("Width 100 Test Accuracy: " + str(net.test(test_features, test_labels)) + str("\n"))

if __name__ == '__main__':
    print("Homework 5 - Begin\n\n")
    doProb2()
    print("\n\n")
    doProb3()
    print("\n\n")
    doProb4()