import numpy as np

class LinearRegression:
    def __init__(self, learning_rate, epochs) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.rand(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            loss = (1/n_samples) * np.sum((y_pred - y)**2)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.bias * db
            print(loss)
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
X = np.random.randn(100, 1)
y = 3 * X.squeeze() + 2 + np.random.randn(100)
model = LinearRegression(learning_rate=1e-3, epochs=10000)
model.fit(X, y)
predictions = model.predict(X)

import matplotlib.pyplot as plt
plt.scatter(X, y, color="blue")
plt.plot(X, predictions, color="red")
plt.show()