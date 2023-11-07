import numpy as np
import copy
import math

from interface import *

# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """
            # your code here \/
            return parameter - self.lr * parameter_grad
            # your code here /\

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """
            # your code here \/
            updater.inertia = self.lr * parameter_grad + self.momentum * updater.inertia
            return parameter - updater.inertia
            # your code here /\

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, ...)), input values

            :return: np.array((n, ...)), output values

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        self.input = copy.deepcopy(inputs)
        self.input[self.input < 0] = 0
        return self.input
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

            :return: np.array((n, ...)), dLoss/dInputs

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        d_layer_d_input = np.array(grad_outputs, copy=True)
        d_layer_d_input[self.forward_inputs < 0] = 0
        return d_layer_d_input
        # your code here /\


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, d)), output values

                n - batch size
                d - number of units
        """
        # your code here \/
        exp_input = np.exp(inputs - inputs.max(axis=1)[:, None])
        softmax = exp_input / np.sum(exp_input, axis=1)[:, None]
        return softmax
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of units
        """
        # your code here \/
        probs = self.forward_outputs
        return probs * grad_outputs - probs * np.diagonal(probs @ grad_outputs.T)[:, None]
        # your code here /\


# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units

        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_units, = self.input_shape
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name='weights',
            shape=(input_units, output_units),
            initializer=he_initializer(input_units)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_units,),
            initializer=np.zeros
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, c)), output values

                n - batch size
                d - number of input units
                c - number of output units
        """
        # your code here \/
        self.input = inputs
        return np.dot(inputs, self.weights) + self.biases
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of input units
                c - number of output units
        """
        # your code here \/
        d_layer_dx = np.dot(grad_outputs, self.weights.T)
        self.weights_grad = np.dot(self.input.T, grad_outputs)
        self.biases_grad = np.ravel(np.sum(grad_outputs, axis=0))
        return d_layer_dx
        # your code here /\


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((1,)), mean Loss scalar for batch

                n - batch size
                d - number of units
        """
        # your code here \/
        return np.mean(-np.log(y_pred[y_gt == 1] + eps)).reshape(-1,)
        # your code here /\

    def gradient_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((n, d)), dLoss/dY_pred

                n - batch size
                d - number of units
        """
        # your code here \/
        batch_size = y_pred.shape[0]
        grad = np.zeros(y_pred.shape)
        grad[y_gt != 0] = -1 / batch_size
        return grad / np.maximum(y_pred, eps)
        # your code here /\


# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    loss = CategoricalCrossentropy()
    optimizer = SGDMomentum(lr=0.01, momentum=0.8)
    model = Model(loss, optimizer)

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Dense(input_shape=(784,), units=1024))
    model.add(ReLU())
    model.add(Dense(units=512))
    model.add(ReLU())
    model.add(Dense(units=128))
    model.add(ReLU())
    model.add(Dense(units=10))
    model.add(Softmax())

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, batch_size=256, epochs=5, x_valid=x_valid, y_valid=y_valid)

    # your code here /\
    return model


# ============================== 3.3.2 convolve ==============================
def convolve(inputs, kernels, padding=0):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    # !!! Don't change this function, it's here for your reference only !!!
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]

    if os.environ.get('USE_FAST_CONVOLVE', False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)


def convolve_numpy(inputs, kernels, padding):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    # your code here \/
    n, d, ih, iw = inputs.shape
    c, d, kh, kw = kernels.shape
    oh = ih + padding*2 - kh + 1
    ow = iw + padding*2 - kw + 1
    outputs = np.zeros((n, c, oh, ow))
    inputs = np.pad(inputs, ((0,0),(0,0),(padding,padding),(padding,padding)))
    for i in range(oh):
        for j in range(ow):
            outputs[:,:, i, j] = inputs[:,:, i:i+kh, j:j+kw].reshape(n, -1) @ kernels[:,:, ::-1, ::-1].reshape(c, -1).T
    return outputs
    # your code here /\


# =============================== 4.1.1 Conv2D ===============================
class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"

        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name='kernels',
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_channels,),
            initializer=np.zeros
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, c, h, w)), output values

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        # your code here \/
        
        return convolve(inputs, self.kernels, padding=(self.kernel_size-1) // 2) + self.biases[None, :, None, None]
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        # your code here \/
        p = self.kernel_size // 2
        p_inv = self.kernel_size - p - 1
        self.biases_grad = np.ravel(np.sum(grad_outputs, axis=(0, 2, 3)))
        self.kernels_grad = convolve(self.forward_inputs[:,:, ::-1, ::-1].transpose(1, 0, 2, 3), grad_outputs.transpose(1, 0, 2, 3), p).transpose(1, 0, 2, 3)
        dL_dX = convolve(grad_outputs, self.kernels.transpose(1, 0, 2, 3)[:,:, ::-1, ::-1], p_inv)
        return dL_dX
        # your code here /\


# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode='max', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {'avg', 'max'}

        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.forward_idxs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)
        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"

        self.output_shape = (channels, output_h, output_w)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, ih, iw)), input values

            :return: np.array((n, d, oh, ow)), output values

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """
        # your code here \/
        def argmax_lastNaxes(matrix, axises):
            s = matrix.shape
            new_shp = s[:-axises] + (np.prod(s[-axises:]),)
            max_idx = matrix.reshape(new_shp).argmax(-1)
            return np.unravel_index(max_idx, s[-axises:])

        n, d, ih, iw = inputs.shape
        oh, ow = math.ceil(ih/self.pool_size), math.ceil(iw/self.pool_size)
        otputs = np.zeros((n, d, oh, ow))
        self.max_idx = np.zeros(inputs.shape)
        for i in range(oh):
            for j in range(ow):
                block = inputs[:,:, i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size]
                if self.pool_mode == 'avg':
                    otputs[:,:, i, j] = block.mean(axis=(2, 3))
                else:
                    otputs[:,:, i, j] = block.max(axis=(2, 3))
                    argmax = argmax_lastNaxes(block, 2)
                    max_idx_part = self.max_idx[:,:, i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size]
                    for channel in range(d):
                        max_idx_part[np.arange(n), channel, argmax[0][..., channel], argmax[1][..., channel]] = 1
        return otputs
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, oh, ow)), dLoss/dOutputs

            :return: np.array((n, d, ih, iw)), dLoss/dInputs

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """
        # your code here \/        
        grad_layer = np.repeat(grad_outputs, self.pool_size, axis=2)
        grad_layer = np.repeat(grad_layer, self.pool_size, axis=3)
        if self.pool_mode == 'avg':
            grad_layer /= self.pool_size**2
        else:
            grad_layer *= self.max_idx
        return grad_layer
        # your code here /\


# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None

        self.forward_inverse_std = None
        self.forward_centered_inputs = None
        self.forward_normalized_inputs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        self.running_mean = np.zeros((input_channels,))
        self.running_var = np.ones((input_channels,))

        self.beta, self.beta_grad = self.add_parameter(
            name='beta',
            shape=(input_channels,),
            initializer=np.zeros
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name='gamma',
            shape=(input_channels,),
            initializer=np.ones
        )

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, d, h, w)), output values

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        # your code here \/
        if self.is_training:
            mean = inputs.mean(axis=(0, 2, 3))
            self.var = inputs.var(axis=(0, 2, 3))
            self.inputs_norm = (inputs - mean[None,:,None,None]) / np.sqrt(self.var[None,:,None,None] + eps)
            self.running_mean = self.momentum*self.running_mean + (1-self.momentum)*mean
            self.running_var = self.momentum*self.running_var + (1-self.momentum)*self.var
        else:
            self.inputs_norm = (inputs - self.running_mean[None,:,None,None]) / np.sqrt(self.running_var[None,:,None,None] + eps)
        return self.inputs_norm*self.gamma[None,:,None,None] + self.beta[None,:,None,None]
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        # your code here \/
        self.gamma_grad = (self.inputs_norm * grad_outputs).sum(axis=(0, 2, 3))
        self.beta_grad = grad_outputs.sum(axis=(0, 2, 3))
        dL_dXnorm = grad_outputs * self.gamma[None,:,None,None]
        dL_dSigma = (dL_dXnorm*self.inputs_norm).mean(axis=(0, 2, 3))[None,:,None,None]
        dL_dX = (dL_dXnorm - dL_dXnorm.mean(axis=(0, 2, 3))[None,:,None,None] - self.inputs_norm*dL_dSigma) / np.sqrt(self.var[None,:,None,None] + eps)
        return dL_dX
        # your code here /\

# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = (np.prod(self.input_shape),)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, (d * h * w))), output values

                n - batch size
                d - number of input channels
                (h, w) - image shape
        """
        # your code here \/
        self.n, self.d, self.h, self.w = inputs.shape
        return inputs.reshape(self.n, -1)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of units
                (h, w) - input image shape
        """
        # your code here \/
        return grad_outputs.reshape((self.n, self.d, self.h, self.w))
        # your code here /\


# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, ...)), input values

            :return: np.array((n, ...)), output values

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        if self.is_training:
            mask = np.random.uniform(low=0.0, high=1.0, size=inputs.shape)
            self.forward_mask = np.zeros(mask.shape)
            self.forward_mask[mask > self.p] = 1
            outputs = self.forward_mask * inputs
        else:
            outputs = (1 - self.p) * inputs
        return outputs
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

            :return: np.array((n, ...)), dLoss/dInputs

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        return grad_outputs * self.forward_mask
        # your code here /\


# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    loss = CategoricalCrossentropy()
    optimizer = SGDMomentum(lr=0.1)
    model = Model(loss, optimizer)

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)

    model.add(Conv2D(output_channels=32, kernel_size=3, input_shape=(3, 32, 32)))
    model.add(ReLU())
    model.add(BatchNorm())
    model.add(Conv2D(output_channels=32, kernel_size=3))
    model.add(ReLU())
    model.add(BatchNorm())
    model.add(Pooling2D(pool_size=4, pool_mode='max'))
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(units=128))
    model.add(ReLU())
    model.add(Dense(units=10))
    model.add(Softmax())

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, batch_size=32, epochs=5, x_valid=x_valid, y_valid=y_valid)

    # your code here /\
    return model

# ============================================================================
