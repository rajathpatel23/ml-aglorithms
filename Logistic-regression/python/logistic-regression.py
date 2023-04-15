# codereference =https://aetperf.github.io/2020/09/18/Logistic-regression-with-JAX.html

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import jax.numpy as jnp
from jax import grad
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def sigmoid_function(x):
    return 1/(1 + jnp.exp(-x))

def predict(B, W, X):
    return sigmoid_function(jnp.dot(X, W) + B)

def loss_function(B, W, X, outputTrue, eps=1e-14, lmbd=0.1):
    n = outputTrue.size
    outputPred = predict(B, W, X)
    outputPred = jnp.clip(outputPred, eps, 1 - eps)
    lossVal = - jnp.sum(outputTrue * jnp.log(outputPred) + (1 - outputTrue) * jnp.log(1 - outputPred))/n 
    + lmbd*jnp.dot(W, W) + B*B
    return lossVal


def train(X_train, y_train, n_epochs, learning_rate):
    W = 1.0e-5 * jnp.ones(n_feat)
    B = 1.0
    tol = 1e-6
    new_cost = float(loss_function(B, W, X_train, y_train))
    cost_hist = [new_cost]

    for i in range(n_epochs):
        currentB = B
        # print(grad(loss_function, argnums=0)(currentB, W, X_train, y_train))
        B -= learning_rate * grad(loss_function, argnums=0)(currentB, W, X_train_s, y_train)
        W -= learning_rate * grad(loss_function, argnums=1)(currentB, W, X_train_s, y_train)
        new_cost = float(loss_function(B, W, X_train, y_train))
        cost_hist.append(new_cost)
        if i > 20 and i % 10 == 0:
            if jnp.abs(cost_hist[-1] - cost_hist[-20]) < tol:
                print(f'Exited loop at iteration {i}')
                break
        if i % 10 == 0:
            print(f"Loss Value: {new_cost}:: epoch: {i}")
    return W, B

def evaluation(B, W, X_test):
    return predict(B, W, X_test)


if __name__ == '__main__':
    X, y = load_breast_cancer(return_X_y=True)
    n_feat = X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=2342)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    paramsW, paramsB = train(X_train, y_train, n_epochs=1000, learning_rate=5e-2)
    yPred = evaluation(paramsB, paramsW, X_test_s)
    yPred = jnp.array(yPred)
    yPred = jnp.where(yPred < 0.5, yPred, 1.0)
    yPred = jnp.where(yPred >= 0.5, yPred, 0.0)
    print(paramsW, paramsB)
    print(yPred.shape, y_test.shape)
    print(classification_report(y_true=y_test, y_pred=yPred))





