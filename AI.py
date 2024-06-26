import numpy as np

class OptimizerAdam:
    def __init__(self, learning_rate=1.0, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = self.learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1./(1. + self.decay * self.iterations))
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.bias)
            layer.bias_momentums = np.zeros_like(layer.bias)

    ​  # Update momentum with current gradients
    ​layer.weight_momentums ​= ​self.beta_1 ​ * ​ \
            layer.weight_momentums ​+ ​ \
            (​1 ​- ​self.beta_1) ​ * ​layer.dweights
    layer.bias_momentums ​= ​self.beta_1 ​ * ​ \
            layer.bias_momentums ​+ ​ \
            (​1 ​- ​self.beta_1) ​ * ​layer.dbiases
    ​  # Get corrected momentum
    # self.iteration is 0 at first pass
    # and we need to start with 1 here
    ​weight_momentums_corrected ​= ​layer.weight_momentums ​ / ​ \
            (​1 ​- ​self.beta_1 ​ ** ​(self.iterations ​+ ​1​))
    bias_momentums_corrected ​= ​layer.bias_momentums ​ / ​ \
            (​1 ​- ​self.beta_1 ​ ** ​(self.iterations ​+ ​1​))
    ​  # Update cache with squared current gradients
    ​layer.weight_cache ​= ​self.beta_2 ​ * ​layer.weight_cache ​+ ​ \
            (​1 ​- ​self.beta_2) ​ * ​layer.dweights​ ** ​2
    ​layer.bias_cache ​= ​self.beta_2 ​ * ​layer.bias_cache ​+ ​ \
            (​1 ​- ​self.beta_2) ​ * ​layer.dbiases​ ** ​2
    Chapter
    10 - Optimizers - Neural
    Networks
    from Scratch in Python
    65
    ​  # Get corrected cache
    ​weight_cache_corrected ​= ​layer.weight_cache ​ / ​ \
            (​1 ​- ​self.beta_2 ​ ** ​(self.iterations ​+ ​1​))
    bias_cache_corrected ​= ​layer.bias_cache ​ / ​ \
            (​1 ​- ​self.beta_2 ​ ** ​(self.iterations ​+ ​1​))
    ​  # Vanilla SGD parameter update + normalization
    # with square rooted cache
    ​layer.weights ​ += -​self.current_learning_rate ​ * ​
    \
    weight_momentums_corrected ​ / ​ \
            (np.sqrt(weight_cache_corrected) ​+
            ​self.epsilon)
    layer.biases ​ += -​self.current_learning_rate ​ * ​ \
            bias_momentums_corrected ​ / ​ \
            (np.sqrt(bias_cache_corrected) ​+
            ​self.epsilon)
    ​  # Call once after any parameter updates
    ​

    def ​p

    ost_update_params​(​self)
    ​:
    self.iterations ​ += ​1


    def post_update_params(self):
        self.iterations += 1

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
    def __init__(self, learning_rate=1.0, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1./(1. + self.decay * self.iterations))
    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            weight_updates = self.momentum * layer.weight_momentums - self.learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.learning_rate * layer.dbiases
            layer.weight_momentums = weight_updates

        else:
            weight_updates = -self.learning_rate * layer.dweights
            bias_updates = -self.learning_rate * layer.dbiases
        layer.weights += weight_updates
        layer.biases += bias_updates
    def post_update_params(self):
        self.iterations += 1

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
