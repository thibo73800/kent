import random
import numpy as np

def markov_two_site(k, l, π):
    # Input k, either 0 or 1
    # S, stationary prob
    γ = π[l] / π[k]
    if random.uniform(0, 1) < γ:
        # Move from k to l accepted
        k = l
    return k


if __name__ == '__main__':

    RP = np.array([
        [0.5,   0.25,   0.,     0.25,   0.,     0.,     0.,     0.,     0.],
        [0.25,  0.25,   0.25,   0.,     0.25,   0.,     0.,     0.,     0.],
        [0.,    0.25,   0.5,    0.,     0.,     0.25,   0.,     0.,     0.],
        [0.25,  0.,     0.,     0.25,   0.25,   0.,     0.25,   0.,     0.],
        [0.,    0.25,   0.,     0.25,   0.,     0.25,   0.,     0.25,   0.],
        [0.,    0.,     0.25,   0.,     0.25,   0.25,   0.,     0.,     0.25],
        [0.,    0.,     0.,     0.25,   0.,     0.,     0.5,    0.25,   0.],
        [0.,    0.,     0.,     0.,     0.25,   0.,     0.25,   0.25,   0.25],
        [0.,    0.,     0.,     0.,     0.,     0.25,   0.,     0.25,   0.5],
    ])

    P = np.zeros((9, 9))

    N = [[2, 4, 0, 0],
        [3, 5, 1, 0],
        [0, 6, 2, 0],
        [5, 7, 0, 1],
        [6, 8, 4, 2],
        [0, 9, 5, 3],
        [8, 0, 0, 4],
        [9, 0, 7, 5],
        [0, 0, 8, 6]]

    π = [1/18, 1/18, 1/18, 1/9, 1/9, 1/9, 1/6, 1/6, 1/6]
    π_count = np.zeros(9)
    k = 0

    """
    for k in range(0, 9):
        for n in N[k]:
            if n == 0:
                n = k
            else:
                n = n - 1
            rel = RP[n, k] * min(1, (π[n]/π[k]))
            P[n, k] = rel
            print("P(%s->%s) =  %s * %s/%s = %s" % (k+1, n+1, RP[n, k], π[n], π[k], rel))

    for line in P:
        print  ("[ " + ", ".join([str(s) for s in line]) + "],")

    import sys
    sys.exit(0)
    """


    for _ in range(0, 1000000):
        l = random.choice(N[k]) - 1
        if l < 0:
            l = k
        nk = markov_two_site(k, l, π)
        P[k, nk] += 1
        k = nk
        π_count[k] += 1


    #print(π)

    #print(P)
    #print(P.sum(axis=1, keepdims=True))
    P = P / P.sum(axis=1, keepdims=True)
    for line in P:

        print  ("[ " + ", ".join([str(s) for s in line]) + "],")
        #print(",".join())


    print( (π_count / π_count.sum()) )





