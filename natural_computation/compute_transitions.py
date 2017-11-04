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

                #print("index = %s, i2 = %s" % (index, i2))
                #print("min(1, %s/%s)  = %s" % (π[i2],π[index], (min(1, π[i2] / π[index]))))
                print("P(%s -> %s) = %s * min(1, %s/%s)  = %s" % (index + 1, i2 + 1, OP[index, i2], π[i2], π[index],  val))

        P[index, index] = 1 - around_sum
        print("P(%s -> %s) = %s" % (index + 1, index + 1, P[index, index]))

    return P


def markov_two_site(k, l, π):
    # Input k, either 0 or 1
    # S, stationary prob
    γ = min(1, π[l] / π[k])
    if random.uniform(0, 1) < γ:
        # Move from k to l accepted
        k = l
    return k

def last_metropolis_algorithm(N, π, iteration):
    """
        Neightb N
        Desired steady state: π

        Return the last state
    """
    P = np.zeros((9, 9))
    k = 0 # We start at k equal 0

    for _ in range(0, iteration):
        l = random.choice(N[k])
        nk = markov_two_site(k, l, π)
        k = nk
    
    return k


def sample_metropolis_algorithm(N, π, iteration):
    """
        Neightb N
        Desired steady state: π
    """
    π_count = np.zeros(9) # State count
    k = 0 # We start at k equal 0

    for _ in range(0, iteration):
        l = random.choice(N[k])
        nk = markov_two_site(k, l, π)
        k = nk
        π_count[k] += 1

    return π_count



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
    S = compute_stationary(P, 1000)
    # Print the propotion
    print(S[0:3].sum())
    print(S[3:6].sum())
    print(S[6:9].sum())

    print("Matrix:")
    for line in P:
        print(", ".join(str(n) for n in  line))

    print("Task 3 ------------------------------------")
    π_count = np.zeros(9)

    for _ in range(10000):
        k = last_metropolis_algorithm(N, π, 3)
        π_count[k] += 1

    proba = π_count / π_count.sum()
    S = compute_stationary(P, 3)
    std = np.std([proba[0], proba[2], proba[8]])

    print("Proba with metropolis => ", proba)
    print("Proba with matrix multiplication => ", S)

    print("Cell 1: %s +- %s" % (proba[0], std))
    print("Cell 3: %s +- %s" % (proba[2], std))
    print("Cell 9: %s +- %s" % (proba[8], std))

    print("Task 4 ------------------------------------")

    π_count = sample_metropolis_algorithm(N, π, 1000000)

    proba = π_count / π_count.sum()
    S = compute_stationary(P, 1000000)
    std = np.std([proba[0], proba[2], proba[8]])

    print("Proba with metropolis => ", proba)
    print("Proba with matrix multiplication => ", S)

    print("Cell 1: %s +- %s" % (proba[0], std))
    print("Cell 3: %s +- %s" % (proba[2], std))
    print("Cell 9: %s +- %s" % (proba[8], std))

