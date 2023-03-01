from scipy.integrate import solve_ivp
import numpy as np
from tqdm import tqdm

import petrinet
import mass_action

def stochastic_euler(x0, f, dt, num_steps):
  soln = [x0]
  for _ in range(num_steps):
    x = soln[-1]
    eps = np.random.normal()
    x_next = x + dt*(f(x) + eps)
    soln.append(x_next)
  return soln

if __name__ == "__main__":
  import matplotlib.pyplot as plt

  S, I, R = "SIR"
  a, b = "ab"
  net = petrinet.PetriNet(
    species     = [S, I, R],
    transitions = [a, b],
    edges_in  = [(S,a), (I,a), (I,b)],
    edges_out = [(a,I), (a,I), (b,R)]
  )

  f = lambda t, x: np.asarray(mass_action.time_dot(net, x=x, rates=[2.0, 1.0])) + np.random.normal(0, 1, len(x))

  for i in tqdm(range(2)):
    soln = solve_ivp(
      fun=f,
      t_span=[0.0, 10.0],
      y0=[0.99, 0.01, 0.00],
      method="RK45",
      t_eval=np.arange(0, 10, 0.1)
    )

    plt.plot(soln.t, soln.y[0,:], 
      color="green", alpha=0.2,
      label = net.species[0] if i == 0 else None)
    plt.plot(soln.t, soln.y[1,:],
      color="red", alpha=0.2,
      label = net.species[1] if i == 0 else None)
    plt.plot(soln.t, soln.y[2,:],
      color="blue", alpha=0.2,
      label = net.species[2] if i == 0 else None)

  plt.legend()
  plt.savefig("figures/sde_demo.png")

  plt.clf()

  net = petrinet.PetriNet(
    species     = ["prey", "predator", "dead"],
    transitions = ["reproduce", "eat", "die"],
    edges_in  = [
      ("prey", "reproduce"),
      ("predator","eat"),
      ("prey","eat"),
      ("predator", "die")
    ],
    edges_out = [
      ("reproduce","prey"),
      ("reproduce","prey"),
      ("eat","predator"),
      ("eat","predator")
    ]
  )

  f = lambda t, x: np.asarray(mass_action.time_dot(net, x=x, rates=[1.0, 1.0, 0.1])) + np.random.normal(0, 1, len(x))

  for i in tqdm(range(1)):
    soln = solve_ivp(fun=f, t_span = [0.0, 20.0], y0 = [1.6, 0.4, 0.02], method = "RK45", t_eval=np.arange(0, 20, 1))

    plt.plot(soln.t, soln.y[0,:], color="green", alpha=0.2,
      label = net.species[0] if i == 0 else None)
    plt.plot(soln.t, soln.y[1,:], color="red", alpha=0.2,
      label = net.species[1] if i == 0 else None)

  plt.legend()
  plt.savefig("figures/sde_demo2.png")