from petrinet import PetriNet
from mass_action import time_dot

from scipy.integrate import solve_ivp

net = PetriNet(
  species     = ["S", "I", "R"],
  transitions = ["a", "b"],
  edges_in  = [("S","a"), ("I","a"), ("I","b")],
  edges_out = [("a","I"), ("a","I"), ("b","R")]
)

soln = solve_ivp(
  fun=lambda t, x: time_dot(net, x=x, rates=[2.0, 1.0]),
  t_span = [0.0, 10.0],
  y0 = [0.99, 0.01, 0.00],
)

print(soln.message) 
