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

    """
    P = np.array([
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
    """

    P = np.array([
        [ 0.494978505112, 0.254181568585, 0.0, 0.250839926303, 0.0, 0.0, 0.0, 0.0, 0.0],
        [ 0.251327782961, 0.251650757195, 0.246249910285, 0.0, 0.250771549559, 0.0, 0.0, 0.0, 0.0],
        [ 0.0, 0.248500163619, 0.500690833727, 0.0, 0.0, 0.250809002654, 0.0, 0.0, 0.0],
        [ 0.125106482187, 0.0, 0.0, 0.375310479641, 0.249011396957, 0.0, 0.250571641216, 0.0, 0.0],
        [ 0.0, 0.125565295572, 0.0, 0.251643065857, 0.124926949876, 0.24762418521, 0.0, 0.250240503484, 0.0],
        [ 0.0, 0.0, 0.123672462829, 0.0, 0.24950498614, 0.377056557584, 0.0, 0.0, 0.249765993448],
        [ 0.0, 0.0, 0.0, 0.166174342538, 0.0, 0.0, 0.583553583601, 0.250272073861, 0.0],
        [ 0.0, 0.0, 0.0, 0.0, 0.167295129215, 0.0, 0.250384282832, 0.332752906139, 0.249567681814],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.167657035629, 0.0, 0.249230158539, 0.583112805832]
        ])


    #P = P.T
    #for line in P:
    #    print line.sum()
    #import sys
    #sys.exit(0)


    # Initial State
    # [Chance to be in A, Chance to be in B]
    So = np.array([1., 0., 0., 0., 0., 0., 0., 0., 0.])

    # TO find the Stationary Matrix S, we can Solve the sytem: S = P.S
    # Or we can launch a simulation as bellow

    # Number of iteration
    i = 10000
    Sp = So # Sp is the previous state
    for _ in range(i):
        Si = np.dot(Sp, P)
        print ("i = %s Si = %s" % (i, Si))
        Sp = Si

    # Sk approach the Stationary Matrix S
    S = Si
    print ("Stationary Matrix = %s" % S)
    #print ("Matrix p^- (the limiting matrix) = \n%s " % LA.matrix_power(P, i))
    print ("So*p^- = \n%s " % np.dot(So, LA.matrix_power(P, i)))

    print(S[0:3].sum())
    print(S[3:6].sum())
    print(S[6:9].sum())

