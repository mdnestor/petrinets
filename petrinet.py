from typing import List, Tuple

class PetriNet():
  def __init__(self, species: List, transitions: List[Tuple], edges_in: List, edges_out: List[Tuple]):
    """
    A Petri net has:
    - `species`, a list of ints;
    - `transitions`, a list of ints;
    - two lists of (s,t) pairs,
      - `edges_in`
      - `edges_out`
    """
    self.species = species
    self.transitions = transitions
    self.edges_in  = edges_in
    self.edges_out = edges_out