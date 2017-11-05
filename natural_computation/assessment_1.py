#!/usr/bin/python3
# -*- coding: utf-8 -*-

# I used python and Numpy which is a python library to handle matrix and vector operations
import numpy as np
import random

def transition_propability(N, Sc, Sp, π):
    """
        Return the transition probability between the State c and State p (*Sp, *Sc)
        **input: **
            *N: (Matrix) Neighbor table
            *π: (Vector) Steady state
            *Sc: Current state index
            *Sp: Proposal state index
    """
    # probability that a site is chosen as a candidate
    probability_chosen = 1./len(N[Sc])
    # probability that the chosen state is accepted
    probability_accepted = min(1, π[Sp] / π[Sc])
    return probability_chosen* probability_accepted

def compute_transitions(N, π):
    """
        This method is used to generate the Transiton Matix from any given steady state
        **input: **
            *N: (Matrix) Neighbor table
            *π: (Vector) Steady state
        **return: **
            *P: Transiton Matrix generated from π
    """
    # New transition matrix. Init with zeros
    P = np.zeros((9, 9))
    # Go through all states
    for Sc, _ in enumerate(π): # Sc: Current State
        Sps_sum = 0
        for Sp in N[Sc]: # Go through all the neighbors of Sc
            if Sp != Sc:
                # Get the transition propability
                tp = transition_propability(N, Sc, Sp, π)
                # Add this value to the sum of all transition (except the transtion from Sc to Sc)
                Sps_sum += tp
                # Add this value in the matrix
                P[Sc, Sp] = tp
        # Probability to stay at the same place
        P[Sc, Sc] = 1 - Sps_sum
    # Retun the matrix
    return P

def metropolis_algorithm(N, π, iterations):
    """
        Metropolis algorithm
        **input: **
            *N: (Matrix) Neighbor table
            *π: (Vector) Steady state
            *iterations: (Scalar) Number of iterations
        *return: **
            *π (Vector) Probability to be in each state after the number of *iterations
            *Sc (Scalar, index) The current state after the number of *iterations
    """
    # State count
    # This vector is used to store the number of time a state is choosen
    π_count = np.zeros(9)
    Sc = 0 # We start at k equal 0. So the first state (1)
    for _ in range(0, iterations):
        # Choose a random state in the the neighbors of Sc
        Sp = random.choice(N[Sc])
        # probability to be accepted
        if random.uniform(0, 1) < min(1, π[Sp] / π[Sc]):
            Sc = Sp
        # Increment this state
        π_count[Sc] += 1
    # Transform the count vector into probability
    π = π_count / iterations
    return π, Sc

def compute_stationary_vector(P, iterations, start=0):
    """
        Method used to compute the stationary vector of a markov chain given
        the Transition Matrix *P
        **input:
            *P: (Matrix) Transition matrix
            *iterations: (Scalar) Number of iterations
            *start: (Optional) Used to start the markov chain at a given state (0 by default)
    """
    # Create the stationary vector at time 0
    Sp = np.array(np.zeros(P[0].shape))
    # Start the markov chain at this state
    Sp[start] = 1
    for _ in range(iterations):
        # Sp+1 = sp.P
        Sp = np.dot(Sp, P) # Dot product between the vector and the matrix (s.P)
    # Return the stationary vector
    return Sp


if __name__ == '__main__':
    N = [[1, 3, 0, 0],  # From state 1, we can in state 2(1+1), 4(3+1), and 1 (0 + 1)
        [2, 4, 0, 1],   # From state 2, we can in state 3(2+1), 5(4+1), 1 (0+1) and 2(1+2)
        [2, 5, 1, 2],   # ...
        [4, 6, 3, 0],
        [5, 7, 3, 1],
        [5, 8, 4, 2],
        [7, 6, 6, 3],
        [8, 7, 6, 4],
        [8, 8, 7, 5]]
    # Stationary states ((1/18)*3 = 1/6), ((1/9)*3 = 2/6), ((1/6)*3 = 1/3)
    π = [1/18, 1/18, 1/18, 1/9, 1/9, 1/9, 1/6, 1/6, 1/6]

    ###
    # Task 2
    ###

    # P is the transition matrix for the Stationaryp states π
    P = compute_transitions(N, π)
    # S is the steady state vector after n iterations
    # If P is well compute, S would be equal to π
    S = compute_stationary_vector(P, 10000)
    # Print the proportion
    print("Task 2:")
    print("S[0:3].sum() = ", S[0:3].sum()) # 1/6
    print("S[3:6].sum() = ", S[3:6].sum()) # 2/6
    print("S[6:9].sum() = ", S[6:9].sum()) # 1/2

    ###
    # Task 3
    ###

    # Vector used to count the probability after 3 iterations
    # Used to compute the standard deviation layer

    print("S with matrix product", compute_stationary_vector(P, 3))
    π_count = np.zeros(9)
    probas_list = np.zeros(9).reshape(1, 9)
    for _ in range(10000):
        # Get the choosen state after 3 steps
        proba, Sc = metropolis_algorithm(N, π, 3)
        π_count[Sc] += 1
        probas_list = np.concatenate((probas_list, [proba]), axis=0)
    π_after_3 = π_count / 10000 # Divided each value in the vector by the number of repetitions
    print("Task 3:")
    # std() is used to get the standard deviation
    print("Cell 1: %s +- %s" % (π_after_3[0], probas_list[:,0].std()))
    print("Cell 3: %s +- %s" % (π_after_3[2], probas_list[:,2].std()))
    print("Cell 9: %s +- %s" % (π_after_3[8], probas_list[:,8].std()))

    ###
    # Task 4
    ###
    # Used to compute the standar deviation layer
    probas_list = np.zeros(9).reshape(1, 9)
    for _ in range(20): # Launch the algorithm 20 time to get the standard deviation
        # Get the choosen state after 3 steps
        proba, Sc = metropolis_algorithm(N, π, 1000000)
        probas_list = np.concatenate((probas_list, [proba]), axis=0)

    print("Task 4:")
    # std() is used to get the standar deviation
    print("Cell 1: %s +- %s" % (probas_list[:,0].mean(), probas_list[:,0].std()))
    print("Cell 3: %s +- %s" % (probas_list[:,2].mean(), probas_list[:,2].std()))
    print("Cell 9: %s +- %s" % (probas_list[:,8].mean(), probas_list[:,8].std()))
