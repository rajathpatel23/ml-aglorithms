import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import classification_report


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=50000, fit_intercept=True, verbose=True) -> None:
        self.learning_rate = learning_rate
        self.num_iteration = num_iterations
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def __b_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    

    def __sigmoid_function(self, z):
        return 1 /(1 + np.exp(-z))
    
    def __loss(self, y, z):
        return (-y * np.log(z) - (1 - y) * np.log(1 - z)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__b_intercept(X)
        self.W = np.random.rand(X.shape[1])

        for i in range(self.num_iteration):
            z = np.dot(X, self.W)

            z = self.__sigmoid_function(z)

            loss = self.__loss(y, z)

            gradient = np.dot(X.T, (z - y)) / y.size

            self.W -= self.learning_rate * gradient

            if(self.verbose ==True and i % 10000 == 0):
                print(f'loss: {loss} \t')
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__b_intercept(X)
        return self.__sigmoid_function(np.dot(X, self.W))
    
    def predict(self, X):
        return self.predict_prob(X).round()

if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = (iris.target != 0) * 1
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
    plt.legend();
    model = LogisticRegression(learning_rate=0.1, num_iterations=300000)
    model.fit(X, y)
    preds = model.predict(X)
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
    plt.legend()
    x1_min, x1_max = X[:,0].min(), X[:,0].max(),
    x2_min, x2_max = X[:,1].min(), X[:,1].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    probs = model.predict_prob(grid).reshape(xx1.shape)
    plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='black');
    plt.show()
    print(classification_report(y, model.predict(X)))
