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
#If you'd like to see the image
#some_digit_image = some_digit.reshape(28, 28)
#plt.imshow(some_digit_image, cmap="binary")
#plt.axis("off")
#plt.show()
#Split into training and test data sets
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
#Train a binary classifier
from sklearn.linear_model import SGDClassifier
#Create target vectors that are boolean; true for 5 and false for not 5
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
#build the classifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
print(sgd_clf.predict([some_digit]))
#Adding the never 5 class
from sklearn.base import BaseEstimator


class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        return self
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


never_5_clf = Never5Classifier()
from sklearn.model_selection import cross_val_score
scores = cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print("Scores:", scores)
#Plot ROC curve
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
from sklearn.ensemble import RandomForestClassifier
#Try out a random forest classifier
forest_clf = RandomForestClassifier(random_state=42)

y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)
plt.plot(fpr, tpr, "b:", label="SGD")


# def plot_roc_curve(fpr, tpr, label=None):
#     plt.plot(fpr, tpr, linewidth=2, label=label)
#     plt.plot([0, 1],[0, 1], 'k--')
#
#
# plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
# plt.legend(loc="lower right")
# plt.show
