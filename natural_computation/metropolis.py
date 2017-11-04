import random
from detailed_balance import OP, compute_stationary
import numpy as np

def markov_two_site(k, l, π):
    # Input k, either 0 or 1
    # S, stationary prob
    γ = min(1, π[l] / π[k])
    if random.uniform(0, 1) < γ:
        # Move from k to l accepted
        k = l
    return k

def metropolis_algorithm(N, π):
    """
        Neightb N
        Desired steady state: π
    """
    # New transition matrix
    P = np.zeros((9, 9))
    π_count = np.zeros(9) # State count
    k = 0 # We start at k equal 0

    for _ in range(0, 50000):
        l = random.choice(N[k])
        nk = markov_two_site(k, l, π)
        P[k, nk] += 1
        k = nk
        π_count[k] += 1
    # Translate rates to probability
    P = P / P.sum(axis=1, keepdims=True)

    return P, π_count


def sample_metropolis_algorithm(N, π):
    """
        Neightb N
        Desired steady state: π
    """
    # New transition matrix
    P = np.zeros((9, 9))
    π_count = np.zeros(9) # State count
    k = 0 # We start at k equal 0

    for _ in range(0, 50000):
        l = random.choice(N[k])
        nk = markov_two_site(k, l, π)
        P[k, nk] += 1
        k = nk
        π_count[k] += 1
    # Translate rates to probability
    P = P / P.sum(axis=1, keepdims=True)

    return P, π_count



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
    P, π_count = metropolis_algorithm(N, π)
    # Compute stationary
    S = compute_stationary(P, 2000)
    # Print the propotion
    print(S[0:3].sum())
    print(S[3:6].sum())
    print(S[6:9].sum())
    print("Matrix:")
    for line in P:
        print(", ".join(str(n) for n in  line))
