import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp  
from sklearn.model_selection import train_test_split

def mse(predict, target):
    return jnp.mean((predict - target) ** 2)

def mse_der(predict, target):
    return 2*(predict - target)/predict.size

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

def mse_der(predict, target):
    return 2*(predict - target)/predict.size

#LeCun initialization
def create_layers(
    network_input_size,
    layer_output_sizes,
    method="LeCun",
):
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

def backpropagation(
    input, layers, activation_funcs, target, activation_ders, cost_der=mse_der, method = "plain", lbd = 0
):
    layer_inputs, zs, predict = feed_forward_saver(input, layers, activation_funcs)

    layer_grads = [() for layer in layers]

    # We loop over the layers, from the last to the first
    for i in reversed(range(len(layers))):
        layer_input, z, activation_der = layer_inputs[i], zs[i], activation_ders[i]

        if i == len(layers) - 1:
            # For last layer we use cost derivative as dC_da(L) can be computed directly
            dC_da = cost_der(predict, target) #<- ChatGPT fikset koden her.             
        else:
            # For other layers we build on previous z derivative, as dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
            (W, b) = layers[i + 1]
            dC_da = dC_dz@W.T

        dC_dz = dC_da*activation_der(z)
        dC_dW = layer_input.T @ dC_dz
        dC_db = np.sum(dC_dz, axis = 0)

        lbd_W, _ = layers[i] #<- ChatGPT hjalp med å legge til denne

        if method == "Lasso":
            dC_dW += lbd*np.sign(lbd_W)
        elif method == "Ridge":
            dC_dW += 2*lbd*lbd_W 

        layer_grads[i] = (dC_dW, dC_db)

    return layer_grads

def step_length(t, t0, t1):
    """The schedule of Eq. (4.40)."""
    return t0/(t+t1)



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
        method = "plain",
        lbd = 0
):

    """LLM assisted (Copilot)"""
    momentums = [(np.zeros_like(W), np.zeros_like(b)) for (W, b) in layers]
    rs = [(np.zeros_like(W), np.zeros_like(b)) for (W, b) in layers]
    n_samples = Xtrain.shape[0]

    t = 0 #<- Chatgpt fikste hvorand adam teller tidsteg.

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

            layer_grads = backpropagation(Xbatch, layers, activation_funcs, Ybatch, activation_ders, cost_der = cost_der, method = method, lbd = lbd)

            for i in range(len(layers)):
                dC_dW, dC_db = layer_grads[i]
                m_W, m_b = momentums[i]
                r_W, r_b = rs[i]

                # Update biased first moment estimate
                m_W = beta1 * m_W + (1 - beta1) * dC_dW
                m_b = beta1 * m_b + (1 - beta1) * dC_db

                # Update biased second raw moment estimate
                r_W = beta2 * r_W + (1 - beta2) * (dC_dW ** 2)
                r_b = beta2 * r_b + (1 - beta2) * (dC_db ** 2)

                # Compute bias-corrected first moment estimate
                m_W_hat = m_W / (1 - beta1 ** (t + 1))
                m_b_hat = m_b / (1 - beta1 ** (t + 1))

                # Compute bias-corrected second raw moment estimate
                r_W_hat = r_W / (1 - beta2 ** (t + 1))
                r_b_hat = r_b / (1 - beta2 ** (t + 1))

                # Update parameters
                gamma_t = step_length(t + 1, t0, t1) if use_decaying_step_length else gamma
                layers[i] = (
                    layers[i][0] - gamma_t * m_W_hat / (np.sqrt(r_W_hat) + 1e-8),
                    layers[i][1] - gamma_t * m_b_hat / (np.sqrt(r_b_hat) + 1e-8)
                )

                # Store updated momentums and second moments
                momentums[i] = (m_W, m_b)
                rs[i] = (r_W, r_b)
            t+=1
            
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
        cost_der=mse_der,
        method = "plain",
        lbd = 0
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

            layer_grads = backpropagation(Xbatch, layers, activation_funcs, Ybatch, activation_ders, cost_der = cost_der, method = method, lbd = lbd)

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
        cost_der=mse_der,
        method = "plain",
        lbd = 0
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

            layer_grads = backpropagation(Xbatch, layers, activation_funcs, Ybatch, activation_ders, cost_der = cost_der, method = method, lbd = lbd)

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