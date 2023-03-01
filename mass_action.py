from typing import List
from petrinet import PetriNet
import numpy as np

def edge_count(edge_list, s, t) -> int:
  return len([e for e in edge_list if e == (s,t)])

def in_matrix(net: PetriNet):
  return np.array([[edge_count(net.edges_in, s,t) for t in net.transitions] for s in net.species])

def out_matrix(net: PetriNet):
  return np.array([[edge_count(net.edges_out, s,t) for t in net.transitions] for s in net.species])

def count_matrix(net: PetriNet):
  return out_matrix(net) - in_matrix(net)

def time_dot(net: PetriNet, x: List[float], rates: List[float]):
  X = np.asarray(x)
  k = np.asarray(rates)
  E_in = in_matrix(net)
  E_diff = count_matrix(net)
  X = np.repeat(X[:,np.newaxis], len(net.transitions), axis=1)
  Y = np.prod(X**E_in, axis=0)
  return np.sum(k * Y * E_diff , axis=1)

if __name__ == "__main__":
  S, I, R = "SIR"
  a, b = "ab"
  
  net = PetriNet(
    species=[S, I, R],
    transitions=[a, b],
    edges_in =[(S,a),(I,a),(I,b)],
    edges_out=[(I,a),(R,b)]
  )

  print(time_dot(net, x=[2,3,5], rates=[2,1]))