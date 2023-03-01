from petrinet import PetriNet
import numpy as np
import networkx as nx

def net_to_graph(net: PetriNet):
  G = nx.DiGraph()
  G.add_nodes_from(net.species, bipartite=0)
  G.add_nodes_from(net.transitions, bipartite=1)

  G.add_edges_from(net.edges_in, directed=True)
  G.add_edges_from(net.edges_out, directed=True)

  return G