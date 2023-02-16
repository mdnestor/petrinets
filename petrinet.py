from typing import List, Tuple
Set = List[int]
EdgeSet = List[Tuple[int, int]]

class PetriNet():
  def __init__(self, species: Set, transitions: Set, edges_in: EdgeSet, edges_out: EdgeSet):
    """
    A Petri net has:
    - `species`, a list of ints;
    - `transitions`, a list of ints;
    - `edges_in`, a list of tuples of ints in the form (s,t) where s is in species and t is in transitions
    ` `edges_out`, a list of tuple of ints in the form (t,s) where s is in species and t is in transitions 
    """
    self.S = species
    self.T = transitions
    self.E_in  = edges_in
    self.E_out = edges_out
