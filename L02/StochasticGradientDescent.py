import numpy as np
import matplotlib.pyplot as plt


# ADALINE with Stochastic Gradient Descent Implementation
class AdalineSGD:
    def __init__(self, eta=0.01, n_iter=50, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.w_initialized = False
        self.cost_ = []
        if random_state:
            np.random.seed(random_state)

    def fit(self, X, y):
        """Fit training data."""
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = np.mean(cost)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights."""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data."""
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to zeros."""
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply ADALINE learning rule to update the weights for a single training sample."""
        output = self.net_input(xi)
        error = target - output
        self.w_[1:] += self.eta * xi * error
        self.w_[0] += self.eta * error  # bias update
        cost = 0.5 * error ** 2
        return cost

    def net_input(self, X):
        """Calculate net input."""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation (simply the net input)."""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step."""
        return np.where(self.activation(X) >= 0.0, 1, -1)


# Generate a suitable synthetic dataset
# For instance, create a simple binary classification dataset in 2D.
np.random.seed(1)
X = np.random.randn(100, 2)
# Define labels: if the sum of the features is positive, assign 1; otherwise, -1.
y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)

# Train the ADALINE SGD model
ada = AdalineSGD(eta=0.01, n_iter=20, random_state=1)
ada.fit(X, y)

# Plot the cost function over epochs to see convergence
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.title('ADALINE SGD - Cost Reduction over Epochs')
plt.show()

# Example predictions on new data
new_samples = np.array([[0.5, -0.2], [-1, 1]])
predictions = ada.predict(new_samples)
print("Predictions for new samples:", predictions)
