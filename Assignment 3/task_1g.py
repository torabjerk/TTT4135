#1g Markov model

def markov_model(prob_dict):
    """ Returns the transition and steady state parameters. """
    P = []
    i = 0
    for key in prob_dict:
        P[i] = []
