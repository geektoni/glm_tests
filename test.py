import pandas as pd
import numpy as np
from pyglmnet import GLM, simulate_glm

import scipy.sparse as sps

#train_x = pd.read_csv("negative_binomial_dataset/1000/X_train_1000_10.csv", header=None)
#train_y = pd.read_csv("negative_binomial_dataset/1000/Y_train_1000_10.csv", header=None)
#train_x = train_x.values
#train_y = train_y.values.flatten()

beta0 = pd.read_csv("negative_binomial_dataset/1000/beta0_1000_10.csv", header=None)
beta = pd.read_csv("negative_binomial_dataset/1000/beta_1000_10.csv", header=None)

beta0 = beta0.values.flatten()
beta = beta.values.flatten()

train_x = np.random.normal(0.0, 1.0, [1000, 10])
train_y = simulate_glm("neg-binomial", beta0, beta, train_x)

glm = GLM(distr="neg-binomial")
glm.fit(train_x, train_y)

print("")
print(glm.beta0_)
print(glm.beta_)
print("")
print(beta0)
print(beta)
