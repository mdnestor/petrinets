from scipy import ndimage

def simulate_reaction_diffusion(x, dx, u0, f, c, dt, num_steps):
  history = [u0]
  k = np.array([[c, -2*c, c]])
  for _ in range(num_steps):
    u = history[-1]
    fu = np.apply_along_axis(f, 0, u)
    u_next = u + dt*(ndimage.convolve(u, k, mode="wrap") + fu)
    history.append(u_next)
  return history

net = PetriNet(
  species     = ["S", "I", "I2", "R"],
  transitions = ["a", "b", "c", "d"],
  edges_in  = [("S","a"), ("I","a"), ("I","b"), ("S", "c"), ("I2", "c"), ("I2", "d")],
  edges_out = [("a","I"), ("a","I"), ("b","R"), ("c", "I2"), ("c", "I2"), ("d", "R")]
)


def u0(x):
  return np.array([
    1.0 - bump((x-.25)/10.)*.1 - bump((x-.75)/.1)*.1,
    bump((x-.25)/.1)*.1,
    bump((x-.75)/.1)*.1,
    0.0*np.heaviside(0.1 - np.abs(x - 0.1), 1.0),
  ])


out = simulate_reaction_diffusion(x, dx, u0(x), lambda x: np.array(time_dot(net, x, rates=[1.0, 0.2, 1.0, 0.2])), c=1, dt=0.1, num_steps=300)

# plot 1

import matplotlib.pyplot as plt
u = out[300]
fig = plt.figure()
new_plot = fig.add_subplot(111)
for i, s in enumerate(net.species):
  new_plot.plot(x, u[i], label=s)
fig.canvas.draw()

# make gif
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

frames = []
fig = plt.figure()
for i, u in tqdm(enumerate(out)):
  fig.clf()
  new_plot = fig.add_subplot(111)
  for i, s in enumerate(net.species):
    new_plot.plot(x, u[i], label=s)

  #plt.ylim(-0.1,1.1)

  fig.canvas.draw()
  image = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
  frames.append(image)

frames[0].save("out.gif", format="GIF", append_images=frames, save_all=True, duration=50, loop=0)
