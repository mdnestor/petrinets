from typing import List
from petrinet import PetriNet
import numpy as np

def time_dot(net: PetriNet, x: List[float], rates: List[float]):
    S = net.species
    T = net.transitions
    E_in  = net.edges_in
    E_out = net.edges_out
    k = rates

    S_idx = {s: i for (i, s) in enumerate(S)}
    T_idx = {t: i for (i, t) in enumerate(T)}
    
    terms = np.array([k[T_idx[t0]] * np.prod([x[(S_idx[s])] for (s,t) in E_in if t == t0]) for t0 in T])

    count_in_list = lambda S, x: len([y for y in S if x == y])
    count = lambda s, t: count_in_list(E_out, (t, s)) - count_in_list(E_in, (s, t))

    D = np.array([[count(s, t) for s in S] for t in T])
    x_dot = np.transpose(D) @ terms
    return x_dot.tolist()
