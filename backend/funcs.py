import numpy as np
from numba import jit


@jit(nopython = True)
def relu(x):
	return np.maximum(0, x)


@jit(nopython = True)
def relu_prime(x):
	return (np.sign(x) + 1.) / 2

@jit(nopython = True)
def linear(x):
	return x


@jit(nopython = True)
def logistic(x):
	return 1. / (1. + np.exp(-x))


@jit(nopython = True)
def logistic_prime(x):
	return logistic(x) * (1 - logistic(x))


@jit(nopython = True)
def softmax(x):
	val = np.exp(x - 1.) / np.sum(np.exp(x - 1.))
	return val


def matrix_mix(A, B, mix_A_prob = 0.8):
	C = np.zeros((len(A), len(A[0])))
	for i in range(len(A)):
		for j in range(len(A[0])):
			if np.random.random() < mix_A_prob:
				C[i][j] = A[i][j]
			else:
				C[i][j] = B[i][j]
	return C


if __name__ == "__main__":
	A = np.ones((5, 5))
	B = np.zeros((5,5))

	print(matrix_mix(A, B))
