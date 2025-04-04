import nn
import numpy as np
import jax.numpy as jnp



def one_hot_encoding(y, num):
        return jnp.eye(num)[y]

def to_one_hot(y_pred, num_classes):
    index = jnp.argmax(y_pred)  # Obtém o índice da maior probabilidade
    res = jnp.zeros_like(y_pred)
    res = res.at[index].set(1)
    return res
if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split


    iris = datasets.load_iris()

    data = iris['data']
    targets = iris['target']
    X_train, X_test, y_train, y_test = train_test_split(

        data, targets, test_size=0.33, random_state=2909)

    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

#    print(X_train)

    y_train = one_hot_encoding(y_train, 3)
    y_test = one_hot_encoding(y_test, 3)
    #print(y_train)
    params = nn.fit([2909, [4, 8, 3], 5000, 0.01], jnp.array(X_train), jnp.array(y_train))
    y_pred = [nn.predict(jnp.array(x), params) for x in X_test]

    getit = 0
    for i, y in enumerate(y_pred):
        print("Valor de X:", X_test[i])
        print("Valor predito: ", y);
        y_hot = to_one_hot(y, 3)
        print("Valor predito em one_hot_encoding: ", y_hot);
        if(jnp.array_equal(y_hot, y_test[i])):
            getit += 1

        print("Valor real: ", y_test[i])

    print("\n\n\n\n\n")
    print("Precisão: ", getit/len(y_pred))



