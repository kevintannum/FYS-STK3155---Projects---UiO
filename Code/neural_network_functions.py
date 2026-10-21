import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp  
from sklearn.model_selection import train_test_split

def mse(predict, target):
    return jnp.mean((predict - target) ** 2)

def mse_der(predict, target):
    return 2*(predict - target)/predict.size

def cross_entropy(predict, target):
    epsilon = 1e-12
    predict = np.clip(predict, epsilon, 1.0 - epsilon,)
    return -np.mean(np.sum(target * np.log(predict),axis=1,))

def cross_entropy_der(predict, target):
    return (predict - target) / predict.shape[0]



def cost(layers, input, activation_funcs, target):
    predict = feed_forward(input, layers, activation_funcs)
    return mse(predict, target)

def ReLU(z):
    return jnp.where(z > 0, z, 0.0)

def ReLU_der(z):
    return jnp.where(z > 0, 1.0, 0.0)

def leaky_ReLU(z):
    return jnp.where(z > 0, z, 0.01*z)

def leaky_ReLU_der(z):
    return jnp.where(z > 0, 1.0, 0.01)

def sigmoid(z):
    return 1 / (1 + jnp.exp(-z))

def sigmoid_der(z):
    return jnp.exp(-z)/((1+jnp.exp(-z))**2)

def softmax(z):
    z = jnp.asarray(z)
    z_shifted = z - jnp.max(z, axis=1, keepdims=True)
    exp_z = jnp.exp(z_shifted)
    return exp_z / jnp.sum(exp_z, axis=1, keepdims=True)


def mse_der(predict, target):
    return 2*(predict - target)/predict.size

def create_layers(network_input_size, layer_output_sizes, method="LeCun",):
    layers = []
    input_size = network_input_size

    for output_size in layer_output_sizes:
        method_lower = method.lower()

        if method_lower == "lecun":
            # Standard deviation: 1 / sqrt(n_inputs)
            scale = np.sqrt(1.0 / input_size)

        elif method_lower == "xavier":
            # Xavier/Glorot normal initialization
            # Standard deviation: sqrt(2 / (n_inputs + n_outputs))
            scale = np.sqrt(2.0 / (input_size + output_size))

        elif method_lower == "he":
            # He normal initialization
            # Standard deviation: sqrt(2 / n_inputs)
            scale = np.sqrt(2.0 / input_size)

        else:
            raise ValueError(
                "Invalid initialization method. "
                "Choose 'LeCun', 'Xavier', or 'He'."
            )

        W = np.random.randn(input_size, output_size) * scale
        b = np.zeros(output_size)

        layers.append((W, b))
        input_size = output_size

    return layers


def feed_forward(input, layers, activation_funcs):
    a = input
    for (W, b), activation_func in zip(layers, activation_funcs):
        z = a @ W + b
        a = activation_func(z)
    return a

def feed_forward_saver(input, layers, activation_funcs):
    layer_inputs = []
    zs = []
    a = input
    for (W, b), activation_func in zip(layers, activation_funcs):
        layer_inputs.append(a)
        z = a@W + b
        a = activation_func(z)

        zs.append(z)

    return layer_inputs, zs, a

#Backpropagation allowing softmax
def backpropagation(
    input,
    layers,
    activation_funcs,
    target,
    activation_ders,
    cost_der=mse_der,
):
    layer_inputs, zs, predict = feed_forward_saver(
        input,
        layers,
        activation_funcs,
    )

    layer_grads = [() for _ in layers]

    # Loop from the output layer to the first hidden layer
    for i in reversed(range(len(layers))):
        layer_input = layer_inputs[i]
        z = zs[i]
        activation_der = activation_ders[i]

        if i == len(layers) - 1:
            if cost_der is cross_entropy_der:
                # Combined softmax + cross-entropy derivative:
                # dC/dz = (prediction - target) / batch size
                dC_dz = cost_der(predict, target)

            else:
                # General case, such as sigmoid + MSE
                dC_da = cost_der(predict, target)
                dC_dz = dC_da * activation_der(z)

        else:
            # Propagate gradient backward from the next layer
            W_next, _ = layers[i + 1]

            dC_da = dC_dz @ W_next.T
            dC_dz = dC_da * activation_der(z)

        dC_dW = layer_input.T @ dC_dz
        dC_db = np.sum(dC_dz, axis=0)
        layer_grads[i] = (dC_dW, dC_db)
    return layer_grads

def step_length(t, t0, t1):
    """The schedule of Eq. (4.40)."""
    return t0/(t+t1)


#Debugged with AI: 
def train_sgd_adam(
    Xtrain,
    Ytrain,
    layers,
    activation_funcs,
    activation_ders,
    epochs,
    batch_size,
    gamma,
    beta1=0.9,
    beta2=0.999,
    use_decaying_step_length=False,
    t0=1.0,
    t1=100.0,
    cost_der=mse_der,
):
    momentums = [
        (
            np.zeros_like(W),
            np.zeros_like(b),
        )
        for W, b in layers
    ]

    rs = [
        (
            np.zeros_like(W),
            np.zeros_like(b),
        )
        for W, b in layers
    ]
    n_samples = Xtrain.shape[0]
    update_step = 0

    for epoch in range(epochs):
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        Xtrain = Xtrain[indices]
        Ytrain = Ytrain[indices]
        for start in range(0, n_samples, batch_size):
            update_step += 1

            end = min(start + batch_size, n_samples)

            Xbatch = Xtrain[start:end]
            Ybatch = Ytrain[start:end]

            layer_grads = backpropagation(
                input=Xbatch,
                layers=layers,
                activation_funcs=activation_funcs,
                target=Ybatch,
                activation_ders=activation_ders,
                cost_der=cost_der,
            )
            for i in range(len(layers)):
                dC_dW, dC_db = layer_grads[i]

                m_W, m_b = momentums[i]
                r_W, r_b = rs[i]

                # First moment
                m_W = beta1 * m_W + (1 - beta1) * dC_dW
                m_b = beta1 * m_b + (1 - beta1) * dC_db

                # Second raw moment
                r_W = beta2 * r_W + (1 - beta2) * dC_dW**2
                r_b = beta2 * r_b + (1 - beta2) * dC_db**2

                # Correct using the number of updates, not epochs
                m_W_hat = m_W / (1 - beta1**update_step)
                m_b_hat = m_b / (1 - beta1**update_step)

                r_W_hat = r_W / (1 - beta2**update_step)
                r_b_hat = r_b / (1 - beta2**update_step)

                if use_decaying_step_length:
                    gamma_t = step_length(
                        update_step,
                        t0,
                        t1,
                    )
                else:
                    gamma_t = gamma

                layers[i] = (
                    layers[i][0]
                    - gamma_t
                    * m_W_hat
                    / (np.sqrt(r_W_hat) + 1e-8),

                    layers[i][1]
                    - gamma_t
                    * m_b_hat
                    / (np.sqrt(r_b_hat) + 1e-8),
                )

                momentums[i] = (m_W, m_b)
                rs[i] = (r_W, r_b)
    return layers



def train_sgd_plain(
        Xtrain,
        Ytrain,
        layers,
        activation_funcs,
        activation_ders,
        epochs,
        batch_size,
        gamma,
        use_decaying_step_length=False,
        t0=1.0,
        t1=100.0,
        cost_der=mse_der
):

    """LLM assisted (Copilot)"""
    n_samples = Xtrain.shape[0]

    for epoch in range(epochs):
        # Shuffle the data at the beginning of each epoch
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        Xtrain = Xtrain[indices]
        Ytrain = Ytrain[indices]

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            Xbatch = Xtrain[start:end]
            Ybatch = Ytrain[start:end]

            layer_grads = backpropagation(Xbatch, layers, activation_funcs, Ybatch, activation_ders, cost_der = cost_der)

            for i in range(len(layers)):
                dC_dW, dC_db = layer_grads[i]

                # Update parameters
                gamma_t = step_length(epoch + 1, t0, t1) if use_decaying_step_length else gamma
                layers[i] = (
                    layers[i][0] - gamma_t * dC_dW,
                    layers[i][1] - gamma_t * dC_db
                )
    
    return layers



def train_sgd_RMSprop(
        Xtrain,
        Ytrain,
        layers,
        activation_funcs,
        activation_ders,
        epochs,
        batch_size,
        gamma,
        beta2=0.999,
        use_decaying_step_length=False,
        t0=1.0,
        t1=100.0,
        cost_der=mse_der
):

    """LLM assisted (Copilot)"""
    momentums = [(np.zeros_like(W), np.zeros_like(b)) for (W, b) in layers]
    rs = [(np.zeros_like(W), np.zeros_like(b)) for (W, b) in layers]
    n_samples = Xtrain.shape[0]

    for epoch in range(epochs):
        # Shuffle the data at the beginning of each epoch
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        Xtrain = Xtrain[indices]
        Ytrain = Ytrain[indices]

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            Xbatch = Xtrain[start:end]
            Ybatch = Ytrain[start:end]

            layer_grads = backpropagation(Xbatch, layers, activation_funcs, Ybatch, activation_ders, cost_der = cost_der)

            for i in range(len(layers)):
                dC_dW, dC_db = layer_grads[i]
                r_W, r_b = rs[i]

                # Update biased second raw moment estimate
                r_W = beta2 * r_W + (1 - beta2) * (dC_dW ** 2)
                r_b = beta2 * r_b + (1 - beta2) * (dC_db ** 2)

                # Update parameters
                gamma_t = step_length(epoch + 1, t0, t1) if use_decaying_step_length else gamma
                layers[i] = (
                    layers[i][0] - gamma_t * dC_dW / (np.sqrt(r_W) + 1e-8),
                    layers[i][1] - gamma_t * dC_db / (np.sqrt(r_b) + 1e-8)
                )

                # Store updated momentums and second moments

                rs[i] = (r_W, r_b)
    return layers


# AI assisted (Copilot) 
def classification_accuracy(layers, X, y, activation_funcs):
    outputs = feed_forward(
        X,
        layers,
        activation_funcs,
    )

    outputs = np.asarray(outputs)
    y = np.asarray(y)

    # Convert network outputs to predicted class labels
    if outputs.ndim == 2 and outputs.shape[1] > 1:
        # Multiclass output, for example shape (n_samples, 10)
        predicted_labels = np.argmax(outputs, axis=1)
    else:
        # Binary output, for example shape (n_samples,) or (n_samples, 1)
        predicted_labels = (outputs.ravel() >= 0.5).astype(int)

    # Convert targets to one integer label per sample
    if y.ndim == 2 and y.shape[1] > 1:
        # One-hot targets, for example shape (n_samples, 10)
        true_labels = np.argmax(y, axis=1)
    else:
        # Integer labels shaped (n_samples,) or (n_samples, 1)
        true_labels = y.ravel().astype(int)

    return np.mean(predicted_labels == true_labels)