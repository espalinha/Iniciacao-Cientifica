import jax
import jax.numpy as jnp

@jax.jit
def softmax(x):
    e_x = jnp.exp(x - jnp.max(x))
    return e_x / e_x.sum(axis=0)

@jax.jit   
def softmax_prime(x):
    return softmax(x) * (1 - softmax(x))
@jax.jit
def relu(x):
  return jnp.maximum(0, x)
@jax.jit
def relu_prime(x):
  return (x > 0).astype(float)
@jax.jit
def identity(x):
  return x
@jax.jit
def identity_prime(x):
  return jnp.ones_like(x)
@jax.jit
def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))
@jax.jit
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))
@jax.jit
def grad_loss(y_pred, y):
  return (y_pred - y)**2
@jax.jit
def grad_loss_prime(y_pred, y):
    return (y_pred - y)
@jax.jit
def binary_cross_entropy(y_true, y_pred, eta=1e-8):
    y_pred = jnp.clip(y_pred, eta, 1.0 - eta)
    return jnp.mean(-(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred)))
@jax.jit
def binary_cross_entropy_prime(y_true, y_pred, eta=1e-5):
    y_pred = jnp.clip(y_pred, eta, 1.0 - eta)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))
@jax.jit
def tanh(x):
    return jnp.tanh(x)
@jax.jit
def tanh_prime(x):
    return 1 - jnp.tanh(x) ** 2

