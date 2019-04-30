import numpy as np
from scipy.special import log1p


def _lmb(beta0, beta, X):
	"""Conditional intensity function."""
	z = beta0 + np.dot(X, beta)
	return _mu(z)

def _mu(z):
	return log1p(np.exp(z))


if __name__ == "__main__":
	
	np.random.seed(42)

	n_samples, n_features = 1000, 10

	beta0 = 1. / (np.float(n_features) + 1.) * \
        np.random.normal(0.0, 1.0)
	beta = 1. / (np.float(n_features) + 1.) * \
        np.random.normal(0.0, 1.0, (n_features,))

	X_train = np.random.normal(0.0, 1.0, [n_samples, n_features])

	mu = _lmb(beta0, beta, X_train)
	theta = 10
	p = mu/(mu+theta)
	Y_train = np.random.negative_binomial(theta, p)

	print(X_train)

	#np.savetxt("beta_{}_{}.csv".format(n_samples, n_features), beta,'%f', delimiter=",")
	with open("beta0_{}_{}.csv".format(n_samples, n_features), "w") as f:
		f.write(str(beta0))
	#np.savetxt("X_train_{}_{}.csv".format(n_samples, n_features), X_train,'%f', delimiter=",")
	#np.savetxt("Y_train_{}_{}.csv".format(n_samples, n_features), Y_train, '%f', delimiter=",")

