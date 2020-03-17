from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
print(mnist.keys())
X, y = mnist["data"], mnist["target"]
print(X.shape)
print(y.shape)
import numpy as np
y = y.astype(np.uint8)
import matplotlib as mpl
import matplotlib.pyplot as plt
some_digit = X[110]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()
#Split into training and test data sets
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
#Train a binary classifier
from sklearn.linear_model import SGDClassifier
y_train_5 = (y_train ==5)
y_test_5 = (y_test ==5)
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
print(sgd_clf.predict([some_digit]))