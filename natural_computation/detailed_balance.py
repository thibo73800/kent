import numpy as np
from numpy import linalg as LA # TO power matrix

# Transition matrix with equal transitions
OP = np.array([
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

# Intuition matrix
IP = np.array([
    [0.25,   0.25,  0.,     0.50,   0.,     0.,     0.,     0.,     0.],
    [1./6,  1./6,   1./6,   0.,     0.50,   0.,     0.,     0.,     0.],
    [0.,    0.25,   0.25,   0.,     0.,     0.50,   0.,     0.,     0.],
    [5./24, 0.,     0.,     5./24,  5./24,  0.,     3./8,   0.,     0.],
    [0.,    5./24,  0.,     5./24,  0.,     5./24,  0.,     3./8,   0.],
    [0.,    0.,     5./24,  0.,     5./24,   5./24,   0.,     0.,     3./8],
    [0.,    0.,     0.,     0.25,   0.,     0.,     0.5,    0.25,   0.],
    [0.,    0.,     0.,     0.,     0.25,   0.,     0.25,   0.25,   0.25],
    [0.,    0.,     0.,     0.,     0.,     0.25,   0.,     0.25,   0.5],
])

def compute_stationary(P, iterations):
    """
        P: Transition matrix
    """
    So = np.array([1., 0., 0., 0., 0., 0., 0., 0., 0.])
    # Number of iteration
    i = iterations
    Sp = So # Sp is the previous state
    for _ in range(i):
        Si = np.dot(Sp, P)
        #print ("i = %s Si = %s" % (i, Si))
        Sp = Si
    # Sk approach the Stationary Matrix S
    S = Si
    print ("Stationary Matrix = %s" % S)
    #print ("Matrix p^- (the limiting matrix) = \n%s " % LA.matrix_power(P, i))
    print ("So*p^- = \n%s " % np.dot(So, LA.matrix_power(P, i)))

    return S

if __name__ == '__main__':
    # First demo of markov chain
    # Using a simple Regular chaines

    # A regular markow chaines have the following points:
    # 1) S.P = S
    # With P the transition matrix and So the state matrix. Sk approach the stationay matrix S
    # 2) Given Any anitial State mateix So, the state matrix Sk approach the Stationay Matrix S
    # 3) The matrix P^k approach a limiting matrix p^- where each row of p^- is equal to
    #    the stationary Matrix S.
    compute_stationary(OP, 20000)

    pass
