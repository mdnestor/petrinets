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
    self.species = species
    self.transitions = transitions
    self.edges_in  = edges_in
    self.edges_out = edges_out
    

def convert_net_to_mat(net):
  S = net.species
  T = net.transitions
  E_in = net.edges_in
  E_out = net.edges_out

  S_idx = {s: i for (s, i) in enumerate(S)}
  T_idx = {t: i for (t, i) in enumerate(T)}

  mat = np.zeros((len(T), len(S), 2), dtype=int)

  for (s, t) in E_in:
    i, j = S_idx[s], T_idx[t]
    mat[j,i][0] += 1
  for (s, t) in E_out:
    i, j = S_idx[s], T_idx[t]
    mat[j,i][1] += 1

  return mat

def convert_mat_to_net(net_mat):

  T = np.arange(net_mat.shape[0])
  S = np.arange(net_mat.shape[1])

  E_in = []
  E_out = []

  for t in T:
    for s in S:
      in_count, out_count = net_mat[t,s]
      for i in range(in_count):
        E_in.append((s,t))
      for i in range(out_count):
        E_out.append((s,t))

  return PetriNet(S, T, E_in, E_out)

net2 = convert_mat_to_net(net_mat)

convert_net_to_mat(net2)



class PetriNet2():
  def __init__(self, n_species: int, n_transitions: int, edges_in: dict, edges_out: dict):
    assert(n_species >= 0 and n_transitions >= 0)

    S = range(n_species)
    T = range(n_transitions)

    dict_items = list(edges_in.items()) + list(edges_out.items())
    assert(all(s in S and t in T and isinstance(v, int) and v > 0 for ((s,t),v) in dict_items))

    self.n_species = n_species
    self.n_transitions = n_transitions
    self.edges_in = edges_in
    self.edges_out = edges_out

def convert_net_to_net2(net: PetriNet) -> PetriNet2:
  return PetriNet2(
    n_species=len(net.species),
    n_transitions=len(net.transitions),
    edge_in = [],
    edges_out = [],
  )

def convert_net2_to_net(net2: PetriNet2) -> PetriNet:
  edges_in = []
  edges_out = []

  for ((s,t),v) in net2.edges_in.items():
    for i in range(v):
      edges_in += [(s,t)]
  for ((s,t),v) in net2.edges_out.items():
    for i in range(v):
      edges_out += [(s,t)]

  return PetriNet(
    species=list(range(net2.n_species)),
    transitions=list(range(net2.n_transitions)),
    edges_in = edges_in,
    edges_out = edges_out,
  )

net2 = PetriNet2(
  n_species=3,
  n_transitions=2,
  edges_in={
    (0,0): 1,
    (1,0): 1,
    (1,1): 1,
  },
  edges_out={
    (1,0): 2,
    (2,1): 1,
  },
)
