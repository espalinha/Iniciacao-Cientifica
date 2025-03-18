import numpy as np

"""
Fonts: 
    + http://neuralnetworksanddeeplearning.com/chap2.html
    + The Elements of Statistical Learning, Data Mining, Inference, and Prediction, Second Edition
    + DeepSeek for optimize the functions and use a pythonic way

    The math behind the model has been learned in Elements of Statistical Learning.

    I used the http://neuralnetworksanddeeplearning.com/chap2.html for understand the backpropagation
    algorithm.

"""

class MultiLayerPerceptron:
    def __init__(self, hidden, epochs, learning_rate=0.001):
        self.hidden = hidden
        self.num_layers = len(hidden)
        self.weights = [np.random.randn(y, x) * np.sqrt(2/x) for x, y in zip(self.hidden[:-1], self.hidden[1:])]
        self.biases = [np.zeros((y, 1)) for y in self.hidden[1:]]  
        self.learning_rate = learning_rate
        self.epochs = epochs

    def relu(self, x):
        return np.maximum(0, x)

    def relu_prime(self, x):
        return (x > 0).astype(float)

    def feed_forward(self, x):
        for i, (b, w) in enumerate(zip(self.biases, self.weights)): #DeepSeek
            x = np.dot(w, x) + b
            if i < len(self.weights) - 1:
                x = self.relu(x)
        return x

    def backpropagation(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.relu(z) if len(activations) < self.num_layers - 1 else z #DeepSeek
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.relu_prime(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
        return nabla_b, nabla_w

    def fit(self, X, Y, batch_size=32):
        for epoch in range(self.epochs):
            indices = np.random.permutation(len(X)) #DeepSeek
            for i in range(0, len(X), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_X = X[batch_indices]
                batch_Y = Y[batch_indices]
                for x, y in zip(batch_X, batch_Y):
                    x = x.reshape(-1, 1)
                    y = y.reshape(-1, 1)
                    nabla_b, nabla_w = self.backpropagation(x, y)

                    self.weights = [w - self.learning_rate * dw for w, dw in zip(self.weights, nabla_w)]
                    self.biases = [b - self.learning_rate * db for b, db in zip(self.biases, nabla_b)]

    def predict(self, X):
        X = np.array(X).reshape(-1, 1)
        return self.feed_forward(X).flatten()

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

if __name__ == "__main__":
    # Gerar dados
#    X = np.random.random(100*1000) * 2 * np.pi
#    y = np.sin(X) + np.cos(2 * X)
    X = [[1, 1], [0, 0], [0, 1], [1, 0]]   
    X = [x for _ in range(10000) for x in X]
    y = [0, 0, 1, 1]
    y = [y_ for _ in range(10000) for y_ in y ]
    
    # Normalizar dados
   
    # Treinar modelo
    mlp = MultiLayerPerceptron([2, 2, 1], epochs=1000, learning_rate=0.01)
    mlp.fit(np.array(X), np.array(y))
    
    # Testar com x = 1.5 (normalizado)
    x_test = [[0, 1], [1, 1], [1, 0], [0, 0]]
    y = [1, 0, 1, 0]
    for i, val in enumerate(x_test):
        y_pred_normalized = mlp.predict(val)
        print("Previsão:", y_pred_normalized)
        print("Valor real:", y[i])
    # Desnormalizar a saída

