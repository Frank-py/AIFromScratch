import numpy as np


class Loss:

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)

        data_loss = np.mean(sample_losses)

        return data_loss


class LossCategorialCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        # bei ln nicht negativ und darf nicht über eins weil negative wahrscheinlichkeit
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            # y_true: hat indexes bsp: [1, 1, 2]
            # samples: [[p1, p2, p3], [p4, p5, p6], [p7, p8, p9]]
            # correct_confidences [[p2], [p5], [p9]]
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        elif len(y_true.shape) == 2:
            # one-hot encoded
            # y_true: hat indexes bsp: [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
            # samples: [[p1, p2, p3], [p4, p5, p6], [p7, p8, p9]]
            # correct_confidences [[p3], [p4], [p8]]
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        labels = len(dvalues)

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # derivative of loss function
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples # normalize gradient


class ActivationSoftmax:

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):

        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # read page 222 for better understanding

            #flatten output array should be one dimensional
            single_output = single_output.reshape(-1, 1)
            # derivative of softmax accounts for every output w.r.t its input
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # summarize into one value again
            self.dinputs = np.dot(jacobian_matrix, single_dvalues)



# this class combines softmax with loss making gradients easier to computer
class ActivationSoftmaxLossCategoricalCrossentropy():
    def __init__(self):
        self.activation = ActivationSoftmax()
        self.loss = LossCategorialCrossentropy()
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    def backward(self, dvalues, y_true):

        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples



class ActivationReLU:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class OptimizerSGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

class LayerDense:

    def __init__(self, n_input, n_neurons):
        self.weights = 0.01 * np.random.randn(n_input, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        self.dinputs = np.dot(dvalues, self.weights.T)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs
