import numpy as np
from numpy import linalg as LA # TO power matrix


if __name__ == '__main__':
    # First demo of markov chain
    # Using a simple Regular chaines

    # A regular markow chaines have the following points:
    # 1) S.P = S
    # With P the transition matrix and So the state matrix. Sk approach the stationay matrix S
    # 2) Given Any anitial State mateix So, the state matrix Sk approach the Stationay Matrix S
    # 3) The matrix P^k approach a limiting matrix p^- where each row of p^- is equal to
    #    the stationary Matrix S.

    # Let's create this similation using Numpy
    # Here two possibile State:
    #   *A: 0.6 chance to stay in A and 0.4 chance to go to B
    #   *B: 0.2 chance to go to A and 0.8 chance to say in B
    P = np.array([
        [0.6, 0.4],
        [0.2, 0.8]
    ])

    # Initial State
    # [Chance to be in A, Chance to be in B]
    So = [0.7, 0.2]

    # TO find the Stationary Matrix S, we can Solve the sytem: S = P.S
    # Or we can launch a simulation as bellow

    # Number of iteration
    i = 30
    Sp = So # Sp is the previous state
    for _ in range(i):
        Si = np.dot(Sp, P)
        print ("i = %s Si = %s" % (i, Si))
        Sp = Si

    # Sk approach the Stationary Matrix S
    S = Si
    print ("Stationary Matrix = %s" % S)
    print ("Matrix p^- (the limiting matrix) = \n%s " % LA.matrix_power(P, i))
    print ("So*p^- = \n%s " % np.dot(So, LA.matrix_power(P, i)))
