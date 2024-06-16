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


class ActivationReLU:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class LayerDense:

    def __init__(self, n_input, n_neurons):
        self.weights = 0.01 * np.random.randn(n_input, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def backward(self, dvalues):
        self.dweights = np.dot(self.weights.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        self.dinputs = np.dot(dvalues, self.weights.T)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
