import random
import numpy as np

def markov_two_site(k, π):
    # Input k, either 0 or 1
    # S, stationary prob
    l = 1 if k == 0 else 0
    γ = π[l] / π[k]
    if random.uniform(0, 1) < γ:
        # Move from k to l accepted
        k = l
    return k


if __name__ == '__main__':

    π = [0.3, 0.8]
    π_count = [0., 0.]
    k = 0

    for _ in range(0, 10):
        k = markov_two_site(k, π)
        π_count[k] += 1

    print(π_count)
    print(π_count[1] / np.sum(π_count))
    print(π_count[0] / np.sum(π_count))
