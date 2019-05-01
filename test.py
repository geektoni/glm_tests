import pandas as pd
import numpy as np
from pyglmnet import GLM, simulate_glm

import scipy.sparse as sps

import seaborn as sns
import matplotlib.pyplot as plt

#train_x = pd.read_csv("negative_binomial_dataset/1000/X_train_1000_10.csv", header=None)
#train_y = pd.read_csv("negative_binomial_dataset/1000/Y_train_1000_10.csv", header=None)
#train_x = train_x.values
#train_y = train_y.values.flatten()

# Read the betas
#beta0 = pd.read_csv("negative_binomial_dataset/10000/beta0_10000_10.csv", header=None)
#beta = pd.read_csv("negative_binomial_dataset/10000/beta_10000_10.csv", header=None)

# Flatten the numpy array. The transformation will have this effect:
# [[a], [b], [c]] ---> [a,b,c]
#beta0 = beta0.values.flatten()
#beta = beta.values.flatten()

np.random.seed(666)

# Generate randomly beta
beta0 = np.random.normal(0.0, 1.0, 1).flatten()
beta = sps.rand(10, 1, 1)
beta = np.array(beta.todense()).flatten()

# Generate random training data by using the previous betas
train_x = np.random.normal(0.0, 1.0, [10000, 10])
train_y = simulate_glm("neg-binomial", beta0, beta, train_x)

# plot the data distribution
sns.set(color_codes=True)
sns.distplot(train_y)
plt.show()

# Create the GLM and train it
glm = GLM(distr="neg-binomial", max_iter=10000)
glm.fit(train_x, train_y)

# Print the betas and the beta0 to check for correctness
print("")
print(glm.beta0_)
print(glm.beta_)
print("")
print(beta0)
print(beta)

# Generate test data
# simulate testing data
X_test = np.random.normal(0.0, 1.0, [1000, 10])
y_test = simulate_glm("poisson", beta0, beta, X_test)

# predict using fitted model on the test data
yhat_test = glm.predict(X_test)

# score the model
deviance = glm.score(X_test, y_test)
print(deviance)
