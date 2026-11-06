"""Unit tests for the FFNN implementation. Run with:  pytest tests/"""
import numpy as np
import pytest
import jax
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression

from src.neural_network_functions import (
    create_layers, feed_forward, backpropagation,
    mse, mse_der, cross_entropy, cross_entropy_der,
    sigmoid, sigmoid_der, ReLU, ReLU_der, leaky_ReLU, leaky_ReLU_der,
    softmax, train_sgd_adam,
)

jax.config.update("jax_enable_x64", True)   # float32 is far too coarse for a 1e-6 gradient check

identity, identity_der = (lambda z: z), (lambda z: np.ones_like(z))
ACTIVATIONS = {"sigmoid": (sigmoid, sigmoid_der),
               "relu": (ReLU, ReLU_der),
               "leaky_relu": (leaky_ReLU, leaky_ReLU_der)}


def _cost(layers, X, acts, Y, cost_name, method, lbd):
    pred = feed_forward(X, layers, acts)
    c = mse(pred, Y) if cost_name == "mse" else cross_entropy(np.asarray(pred), Y)
    if method == "Ridge":
        c += lbd * sum(np.sum(W**2) for W, _ in layers)
    if method == "Lasso":
        c += lbd * sum(np.sum(np.abs(W)) for W, _ in layers)
    return float(c)


@pytest.mark.parametrize("method", ["plain", "Ridge", "Lasso"])
@pytest.mark.parametrize("cost_name", ["mse", "cce"])
@pytest.mark.parametrize("act_name", ACTIVATIONS)
def test_backprop_matches_finite_difference(act_name, cost_name, method, h=1e-5, lbd=0.1):
    """(i) analytical backprop gradient == central finite difference, to 1e-6."""
    rng = np.random.default_rng(1)
    n, p, K = 8, 3, 4
    X = rng.normal(size=(n, p))
    act, act_der = ACTIVATIONS[act_name]
    np.random.seed(0)
    layers = create_layers(p, [5, K])
    if cost_name == "mse":
        Y, acts, ders, cder = rng.normal(size=(n, K)), [act, identity], [act_der, identity_der], mse_der
    else:  # softmax output + cross-entropy, as in the classifier
        Y, acts, ders, cder = np.eye(K)[rng.integers(K, size=n)], [act, softmax], [act_der, identity_der], cross_entropy_der

    grads = backpropagation(X, layers, acts, Y, ders, cost_der=cder, method=method, lbd=lbd)

    for i, (W, b) in enumerate(layers):
        for arr, g in ((W, grads[i][0]), (b, grads[i][1])):
            for idx in np.ndindex(arr.shape):
                orig = arr[idx]
                arr[idx] = orig + h; cp = _cost(layers, X, acts, Y, cost_name, method, lbd)
                arr[idx] = orig - h; cm = _cost(layers, X, acts, Y, cost_name, method, lbd)
                arr[idx] = orig
                assert abs((cp - cm) / (2 * h) - g[idx]) < 1e-6, f"layer {i} {idx}"


def test_softmax_sums_to_one():
    """(ii) softmax rows are probabilities, also for large inputs."""
    z = np.random.default_rng(0).normal(scale=50, size=(20, 10))
    P = np.asarray(softmax(z))
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12) and (P >= 0).all()


def test_no_hidden_layer_matches_logistic_regression():
    """(iii) zero hidden layers + softmax + cross-entropy == multinomial logistic regression."""
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=2.5, random_state=0)
    Y = np.eye(3)[y]
    np.random.seed(0)
    layers = create_layers(2, [3])
    layers = train_sgd_adam(X, Y, layers, [softmax], [identity_der], epochs=300,
                            batch_size=32, gamma=0.05, cost_der=cross_entropy_der)
    ours = np.argmax(np.asarray(feed_forward(X, layers, [softmax])), axis=1)
    sk = LogisticRegression(C=1e6, max_iter=2000).fit(X, y).predict(X)
    assert np.mean(ours == sk) >= 0.98
