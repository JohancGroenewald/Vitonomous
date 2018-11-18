import pickle
import sys

import numpy as np
from scipy import signal
from scipy.special import expit as activation_function
from scipy.stats import truncnorm


class Classifications:
    IS_NAC = 0
    IS_CLASS_1 = 1
    IS_CLASS_2 = 2
    IS_CLASS_3 = 3
    IS_CLASS_4 = 4
    IS_CLASS_5 = 5
    IS_CLASS_6 = 6
    IS_CLASS_7 = 7
    IS_CLASS_8 = 8
    IS_CLASS_9 = 9
    IS_CLASS_10 = 10
    IS_CLASS_11 = 11
    IS_CLASS_12 = 12
    IS_CLASS_13 = 13
    IS_CLASS_14 = 14
    IS_CLASS_15 = 15
    IS_CLASS_16 = 16
    IS_CLASS_17 = 17
    IS_CLASS_18 = 18
    IS_CLASS_19 = 19
    IS_CLASS_20 = 20

    @staticmethod
    def name(index):
        return 'NAC' if index == 0 else 'CLASS_{}'.format(index)


class NearestNeighbor:
    FILE_NAME = 'nearest_neighbor.pickle'

    def __init__(self, shape):
        self.shape = shape
        self.Xtr = None
        self.ytr = None
        r, c = self.shape
        self.lin_space = np.linspace(0.1, 0.9, (r*c))

    def __len__(self):
        return 0 if self.Xtr is None else self.Xtr.size

    def prepare(self, data):
        data = data.ravel()
        return data

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        num_test = X.shape[0]
        y_predict = np.zeros(num_test, dtype=self.ytr.dtype)
        for i in range(num_test):
            distance = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            distance = self.softmax(distance)
            max_index = np.argmax(distance)
            if distance[max_index] > 0.5:
                y_predict[i] = self.ytr[max_index]
            else:
                y_predict[i] = 0
        return y_predict

    @staticmethod
    def softmax(x):
        m = np.max(x)
        if m > 0:
            x /= m
            x = 1-x
        return x

    def save(self):
        with open(self.FILE_NAME,'wb') as outfile:
            pickle.dump(self.Xtr, outfile)
            pickle.dump(self.ytr, outfile)

    def load(self):
        with open(self.FILE_NAME,'rb') as infile:
            self.Xtr = pickle.load(infile)
            self.ytr = pickle.load(infile)


class BasicConvolution:
    def __init__(self):
        np.random.seed(598765)

        # 0. Declare Weights
        self.w1 = np.random.randn(2, 2) * 4
        self.w2 = np.random.randn(4, 1) * 4

        # 1. Declare hyper Parameters
        self.num_epoch = 1000
        self.learning_rate = 0.7

        self.cost_before_train = 0
        self.cost_after_train = 0
        self.final_out, self.start_out = np.array([[]]), np.array([[]])

        self.X, self.y = np.array([[]]), np.array([[]])

    def prepare(self, data):
        return data

    def train(self):
        for epoch in range(self.num_epoch):
            for i in range(len(self.X)):
                layer_1 = signal.convolve2d(self.X[i], self.w1, 'valid')
                layer_1_act = self.tanh(layer_1)

                layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act, -1), axis=0)
                layer_2 = layer_1_act_vec.dot(self.w2)
                layer_2_act = self.log(layer_2)

                cost = np.square(layer_2_act-self.y[i]).sum() * 0.5
                print("Current iter : ", epoch, " Current train: ", i, " Current cost: ", cost, end="\r")

                grad_2_part_1 = layer_2_act-self.y[i]
                grad_2_part_2 = self.d_log(layer_2)
                grad_2_part_3 = layer_1_act_vec
                grad_2 = grad_2_part_3.T.dot(grad_2_part_1*grad_2_part_2)

                grad_1_part_1 = (grad_2_part_1*grad_2_part_2).dot(self.w2.T)
                grad_1_part_2 = self.d_tanh(layer_1)
                grad_1_part_3 = self.X[i]

                grad_1_part_1_reshape = np.reshape(grad_1_part_1, (2, 2))
                grad_1_temp_1 = grad_1_part_1_reshape * grad_1_part_2
                grad_1 = np.rot90(signal.convolve2d(grad_1_part_3, np.rot90(grad_1_temp_1, 2), 'valid'), 2)

                self.w2 = self.w2 - grad_2 * self.learning_rate
                self.w1 = self.w1 - grad_1 * self.learning_rate

    def predict(self, X):
        num_test = X.shape[0]
        y_predict = np.zeros(num_test, dtype=self.y.dtype)
        for i in range(len(self.X)):
            layer_1 = signal.convolve2d(self.X[i], self.w1, 'valid')
            layer_1_act = self.tanh(layer_1)

            layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act, -1), axis=0)
            layer_2 = layer_1_act_vec.dot(self.w2)
            layer_2_act = self.log(layer_2)


        return y_predict

    def cost_before_training(self):
        for i in range(len(self.X)):
            layer_1 = signal.convolve2d(self.X[i], self.w1, 'valid')
            layer_1_act = self.tanh(layer_1)

            layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act, -1), axis=0)
            layer_2 = layer_1_act_vec.dot(self.w2)
            layer_2_act = self.log(layer_2)
            cost = np.square(layer_2_act-self.y[i]).sum() * 0.5
            self.cost_before_train = self.cost_before_train + cost
            self.start_out = np.append(self.start_out, layer_2_act)

    def cost_after_training(self):
        for i in range(len(self.X)):
            layer_1 = signal.convolve2d(self.X[i], self.w1, 'valid')
            layer_1_act = self.tanh(layer_1)

            layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act, -1), axis=0)
            layer_2 = layer_1_act_vec.dot(self.w2)
            layer_2_act = self.log(layer_2)
            cost = np.square(layer_2_act-self.y[i]).sum() * 0.5
            self.cost_after_train = self.cost_after_train + cost
            self.final_out = np.append(self.final_out, layer_2_act)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def d_tanh(x):
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def log(x):
        return 1/(1 + np.exp(-1 * x))

    def d_log(self, x):
        return self.log(x) * (1 - self.log(x))


class BasicNeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = self.sigmoid(np.dot(self.input, self.weights1))
        self.output = self.sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(
            self.layer1.T,
            (2*(self.y-self.output)*self.sigmoid_derivative(self.output))
        )
        d_weights1 = np.dot(
            self.input.T,
            (np.dot(2*(self.y-self.output)*self.sigmoid_derivative(self.output), self.weights2.T)*self.sigmoid_derivative(self.layer1))
        )

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    @staticmethod
    def sigmoid(x):
        return 1.0/(1+np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x*(1.0-x)


def truncated_normal(mean=0, sd=1, low: float=0, upp: float=10):
    return truncnorm((low - mean) / sd,
                     (upp - mean) / sd,
                     loc=mean,
                     scale=sd)


class NeuralNetwork:
    """
    network_structure: ie. [input_nodes, hidden1_nodes, ... , hidden_n_nodes, output_nodes]
    """
    def __init__(self, network_structure, learning_rate, bias=None):
        self.structure = network_structure
        self.learning_rate = learning_rate
        self.bias = bias
        self.weights_matrices = []
        self.create_weight_matrices()
        self.factor = 255 * 0.99 + 0.01
        self.len = 0

    def __len__(self):
        return self.len

    def create_weight_matrices(self):
        X = None  # truncated_normal(mean=2, sd=1, low=-0.5, upp=0.5)
        bias_node = 1 if self.bias else 0
        layer_index = 1
        no_of_layers = len(self.structure)
        while layer_index < no_of_layers:
            nodes_in = self.structure[layer_index-1]
            nodes_out = self.structure[layer_index]
            n = (nodes_in + bias_node) * nodes_out
            rad = 1 / np.sqrt(nodes_in)
            X = truncated_normal(mean=2, sd=1, low=-rad, upp=rad)
            wm = X.rvs(n).reshape((nodes_out, nodes_in + bias_node))
            self.weights_matrices.append(wm)
            layer_index += 1

    def prepare(self, data):
        data = data.ravel()
        data = data / self.factor
        return data

    def train_single(self, input_vector, target_vector):
        # input_vector and target_vector can be tuple, list or ndarray
        no_of_layers = len(self.structure)
        input_vector = np.array(input_vector, ndmin=2).T
        layer_index = 0
        # The output/input vectors of the various layers:
        res_vectors = [input_vector]
        out_vector = None
        while layer_index < no_of_layers - 1:
            in_vector = res_vectors[-1]
            if self.bias:
                # adding bias node to the end of the 'input'_vector
                in_vector = np.concatenate((in_vector, [[self.bias]]))
                res_vectors[-1] = in_vector
            x = np.dot(self.weights_matrices[layer_index], in_vector)
            out_vector = activation_function(x)
            res_vectors.append(out_vector)
            layer_index += 1

        layer_index = no_of_layers - 1
        target_vector = np.array(target_vector, ndmin=2).T
        # The input vectors to the various layers
        output_errors = target_vector - out_vector
        while layer_index > 0:
            out_vector = res_vectors[layer_index]
            in_vector = res_vectors[layer_index-1]
            if self.bias and not layer_index==(no_of_layers-1):
                out_vector = out_vector[:-1,:].copy()
            tmp = output_errors * out_vector * (1.0 - out_vector)
            tmp = np.dot(tmp, in_vector.T)
            #if self.bias:
            #    tmp = tmp[:-1,:]
            self.weights_matrices[layer_index-1] += self.learning_rate * tmp
            output_errors = np.dot(self.weights_matrices[layer_index-1].T, output_errors)
            if self.bias:
                output_errors = output_errors[:-1,:]
            layer_index -= 1

    def train(self, data_array, labels_one_hot_array, labels, epochs=1, break_out=0.5):
        self.len = 1
        for epoch in range(epochs):
            for i in range(len(data_array)):
                self.train_single(data_array[i], labels_one_hot_array[i])
            corrects, wrongs = self.evaluate(data_array, labels)
            accruracy = corrects/(corrects+wrongs)
            print("\rAccruracy: {}, {}/{}".format(accruracy, epoch, epochs), end='', flush=True)
            # if epoch > 10 and accruracy > break_out:
            #     break
        print()

    def predict(self, input_vector):
        # input_vector can be tuple, list or ndarray
        no_of_layers = len(self.structure)
        if self.bias:
            # adding bias node to the end of the inpuy_vector
            input_vector = np.concatenate((input_vector, [self.bias]))
        in_vector = np.array(input_vector, ndmin=2).T
        layer_index = 1
        # The input vectors to the various layers
        out_vector = None
        while layer_index < no_of_layers:
            x = np.dot(self.weights_matrices[layer_index-1], in_vector)
            out_vector = activation_function(x)
            # input vector for next layer
            in_vector = out_vector
            if self.bias:
                in_vector = np.concatenate((in_vector, [[self.bias]]))
            layer_index += 1
        return out_vector

    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.predict(data[i])
            res_max = res.argmax()
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs

    def save(self):
        pass

    def load(self):
        pass
