from petrinet import PetriNet
from mass_action import time_dot

import numpy as np
from scipy import ndimage

def simulate_reaction_diffusion(x, dx, u0, f, c, dt, num_steps):
  soln = [u0]
  k = np.array([[c, -2*c, c]])
  for _ in range(num_steps):
    u = soln[-1]
    fu = np.apply_along_axis(f, 0, u)
    u_next = u + dt*(ndimage.convolve(u, k, mode="wrap") + fu)
    soln.append(u_next)
  return np.asarray(soln)

# example
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

net = PetriNet(
  species     = ["S", "I", "R"],
  transitions = ["a", "b"],
  edges_in  = [("S","a"), ("I","a"), ("I","b")],
  edges_out = [("a","I"), ("a","I"), ("b","R")]
)

dx = 0.01
x = np.arange(0, 1, dx)

bump = np.vectorize(lambda x: np.exp(1 - 1/(1-x**2)) if np.abs(x) < 1 else 0.0)

def u0(x):
  return np.array([
    1.0 - bump((x-.25)/10.)*.1,
    bump((x-.25)/.1)*.1,
    0.0*np.heaviside(0.1 - np.abs(x - 0.1), 1.0),
  ])

soln = simulate_reaction_diffusion(x, dx, u0(x), lambda x: np.array(time_dot(net, x, rates=[1.2, 0.1])), c=1, dt=0.1, num_steps=1000)

# save result to gif
frames = []
fig = plt.figure()
for i, u in tqdm(enumerate(soln)):
  fig.clf()
  new_plot = fig.add_subplot(111)
  for i, s in enumerate(net.species):
    new_plot.plot(x, u[i], label=s)

  fig.canvas.draw()
  image = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
  frames.append(image)

frames[0].save("soln.gif", format="GIF", append_images=frames, save_all=True, duration=50, loop=0)

# heat maps
n_species = len(net.species)
fig, axs = plt.subplots(1, n_species, figsize=(3*n_species, 6))
for i in range(n_species):
  axs[i].imshow(soln[:,i,:], aspect=0.5, cmap="Blues")
fig.show()
