import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import petrinet
import mass_action

S, I, R = "SIR"
a, b = "ab"
net = petrinet.PetriNet(
  species=[S, I, R],
  transitions=[a, b],
  edges_in =[(S,a),(I,a),(I,b)],
  edges_out=[(I,a),(I,a),(R,b)]
)

f = lambda t, x: mass_action.time_dot(net, x=x, rates=[2.0, 1.0])

soln = solve_ivp(
  fun=f,
  t_span=[0.0, 10.0],
  y0=[0.99, 0.01, 0.00],
  method="RK45"
)
print(soln.message)

for i in range(len(net.species)):
  plt.plot(soln.t, soln.y[i,:], label=net.species[i])
plt.legend()
plt.savefig("figures/ode_demo.png")