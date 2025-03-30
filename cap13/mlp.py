import jax
import jax.numpy as jnp
from numpy import gradient
import function as fn
from typing import NamedTuple


#------------------------Structs-----------------------

class Params(NamedTuple):
    weights: list
    bias: list

class Attributes(NamedTuple):
    key: jax.Array
    hidden: list
    epochs: int
    eta: float
    size: int

class ParamsGrads(NamedTuple):
    weights_grad: list
    bias_grad: list

class ReturnType(NamedTuple):
    params: Params
    attr: Attributes
#------------------------Structs-----------------------


# ----------------------------------
def __init_params(key, hidden, epochs, eta):
    __key = jax.random.PRNGKey(key)        
    __hidden = hidden
    __epochs = epochs
    __eta = eta
    __size = len(hidden)
    attr = Attributes(__key, __hidden, __epochs, __eta, __size)
    keys = jax.random.split(__key, len(hidden) - 1)
    weights = [jax.random.normal(key=k, shape=(y, x)) * jnp.sqrt(2/x) for k, (x, y) in zip(keys, zip(hidden[:-1], hidden[1:]))]
    biases = [jnp.zeros(shape=(y,)) for y in hidden[1:]]
    __params = Params(weights, biases)
    return ReturnType(__params, attr)

def __grad_loss(y_pred, y):
    #return fn.binary_cross_entropy(y, y_pred)
    return fn.grad_loss(y_pred, y)
@jax.jit
def __grad_loss_prime(y_pred, y):
    #return fn.binary_cross_entropy_prime(y, y_pred)
    return fn.grad_loss_prime(y_pred, y)
@jax.jit
def __hidden_func(x):
    #return fn.relu(x)
    return fn.tanh(x)
@jax.jit
def __hidden_func_prime(x):
    return fn.tanh_prime(x)
    #return fn.relu_prime(x)
@jax.jit
def __output_funcs(x):
    #return fn.sigmoid(x)     
    return fn.identity(x)
@jax.jit
def __output_funcs_prime(x):
    #return fn.sigmoid_prime(x)

    return fn.identity_prime(x)
@jax.jit
def __feedforward(x, param: ReturnType) -> jax.Array:
    for i, (w, b) in enumerate(zip(param.params.weights, param.params.bias)):
        x = jnp.matmul(w, x) + b
        if i == len(param.params.weights) - 1:
            x = __output_funcs(x)
        else: x = __hidden_func(x)
    return x   
@jax.jit
def __update_weights(params:ReturnType, params_grad: ParamsGrads):
    """  
    new_weights = []
    for w, dw in zip(params.params.weights, params_grad.weights_grad):
        #jax.debug.print("dw, update {}", dw)
        #jax.debug.print("w, update {}", w)
        new_weights.append(w - params.attr.eta * dw)
    """
    new_weights = jax.tree.map(lambda w, dw: w - params.attr.eta * dw, params.params.weights, params_grad.weights_grad)
    """
    new_bias = []
    for b, db in zip(params.params.bias, params_grad.bias_grad):
        #jax.debug.print("db, update {}", db)
        #jax.debug.print("b, update {}", b)
        new_bias.append(b - params.attr.eta * db)
    """
    new_bias = jax.tree.map(lambda b, db: b - params.attr.eta * db, params.params.bias, params_grad.bias_grad)
    
    return ReturnType(
        params=Params(new_weights, new_bias),
        attr=params.attr
    )
@jax.jit
def __backforward(x, y, params: ReturnType) -> ParamsGrads:
    # Forward pass (guardando ativações)
    x_vec = jnp.reshape(x, (-1,))
    activations = [x_vec]
    zs = []
    
    current_activation = x_vec
    for i, (w, b) in enumerate(zip(params.params.weights, params.params.bias)):
        z = jnp.dot(w, current_activation) + b
        zs.append(z)
        
        if i == len(params.params.weights) - 1:
            current_activation = __output_funcs(z)
        else:
            current_activation = __hidden_func(z)
            
        activations.append(current_activation)
    
    # Backward pass
    #output_error = (activations[-1] - y) * __output_funcs_prime(zs[-1])
    output_error = __grad_loss_prime(activations[-1], y) * __output_funcs_prime(zs[-1])

    
    w_grads = [jnp.zeros_like(w) for w in params.params.weights]
    b_grads = [jnp.zeros_like(b) for b in params.params.bias]
    
    # Gradiente da última camada
    w_grads[-1] = jnp.outer(output_error, activations[-2])
    b_grads[-1] = output_error
    
    # Propagação do erro para trás
    for l in range(2, len(params.params.weights)):
        error = jnp.dot(params.params.weights[-l+1].T, output_error) * __hidden_func_prime(zs[-l])
        w_grads[-l] = jnp.outer(error, activations[-l-1])
        b_grads[-l] = error
        output_error = error
    
    return ParamsGrads(w_grads, b_grads)    

#array: [key, hidden, epochs, eta]
def fit(array, X, y):
    print("initing")
    init_params = __init_params(array[0], array[1], array[2], array[3])
    print("training")
    def epoch_step(epoch, params):
        def batch_step(params, batch):
            x_, y_ = batch
            gradient = __backforward(x_, y_, params)
            return __update_weights(params, gradient)
        
        # Aplicar o gradiente para todo o dataset
        params = jax.lax.fori_loop(0, len(X), lambda i, p: batch_step(p, (X[i], y[i])), params)
        
        loss = jnp.mean(jax.vmap(lambda xx, yy: __grad_loss(__feedforward(xx, params), yy))(X, y))

        return jax.lax.cond(jnp.isclose(loss, 1e-5), lambda: params, lambda: params)
    
    # Loop de épocas sem `for`
    final_params = jax.lax.fori_loop(0, init_params.attr.epochs, epoch_step, init_params)           
    return final_params

def predict(X:jax.Array, params: ReturnType):
    y = __feedforward(X, params)
    return y


if __name__ == "__main__":
    """
    k = [0, 1]
    X = []
    for i in k:
        for j in k:
            X.append(jnp.array([i, j]))
    X = jnp.array(X)
    y = jnp.array([0, 1, 1, 0])
    params = fit([2909, [2, 2, 1], 2000, 0.1], X, y)
    for i, x in enumerate(X):
        print("x: ",x)
        y_pred = predict(x, params)
        print(y_pred)
        print("y_pred: ",jnp.where(y_pred > 0.5, 1, 0))
        print("verdadeiro: ", y[i])
    """
    
    x = jnp.linspace(start=0.01, stop=2*jnp.pi, num=15000)
    x = jnp.reshape(x, (-1, 1))
    y = jnp.sin(x) + jnp.cos(2*x)
    
    params = fit([2909, [1, 32, 8, 8, 32, 1], 4500, 0.01], x, y)
    print("X: ", 1.5)
    print("Valor real: ", jnp.sin(1.5) + jnp.cos(2*1.5))
    print("Valor predito: ", predict(jnp.array([1.5]), params))
    

