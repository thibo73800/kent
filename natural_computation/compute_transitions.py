import random
from detailed_balance import OP, compute_stationary
import numpy as np


def compute_transitions(N, π):
    """
        Neightb N
        Desired steady state: π
    """
    # New transition matrix
    P = np.zeros((9, 9))
    π_count = np.zeros(9) # State count

    for index, value in enumerate(π):
        around_sum = 0
        for i2 in N[index]:
            if index != i2:
                val = 1./4 * min(1, π[i2] / π[index])
                around_sum += val
                P[index, i2] = val

                print("index = %s, i2 = %s" % (index, i2))
                print("min(1, %s/%s)  = %s" % (π[i2],π[index], (min(1, π[i2] / π[index]))))
                print("P(%s -> %s) = %s * min(1, %s/%s)  = %s" % (index + 1, i2 + 1, OP[index, i2], π[i2], π[index],  val))

        P[index, index] = 1 - around_sum

    return P

if __name__ == '__main__':
    N = [[1, 3, 0, 0],
        [2, 4, 0, 1],
        [2, 5, 1, 2],
        [4, 6, 3, 0],
        [5, 7, 3, 1],
        [5, 8, 4, 2],
        [7, 6, 6, 3],
        [8, 7, 6, 4],
        [8, 8, 7, 5]]
    # Desired stationary state
    π = [1/18, 1/18, 1/18, 1/9, 1/9, 1/9, 1/6, 1/6, 1/6]
    P = compute_transitions(N, π)
    # Compute stationary
    S = compute_stationary(P, 2000)
    # Print the propotion
    print(S[0:3].sum())
    print(S[3:6].sum())
    print(S[6:9].sum())
    print("Matrix:")
    for line in P:
        print(", ".join(str(n) for n in  line))
