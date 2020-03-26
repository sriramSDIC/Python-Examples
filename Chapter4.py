#Linear regression
#create data for y=4+3x+w
import numpy as np
#import matplotlib.pyplot as plt
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100,1)
#plt.scatter(X,y)
#plt.xlabel('X')
#plt.ylabel('y')
#plt.show()
#Let us compute the analytical solution for theta, inv(X^T X)X^T y
X_b = np.c_[np.ones((100,1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print("theta_best:", theta_best)


