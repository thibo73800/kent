import numpy as np

if __name__ == '__main__':
	P = np.zeros((9, 9))
	P[0][0] = 1/2
	P[0][1] = 1/4
	P[0][3] = 1/4


	print(P)