from scipy import ndimage
import numpy as np

import petrinet
import mass_action

def reaction_diffusion(x, dx, u0, f, c, dt, n_t):
  soln = [u0]
  k = np.array([[c, 1-2*c, c]])
  for _ in range(n_t):
    u = soln[-1]
    ku = ndimage.convolve(u, k, mode="wrap")
    fu = np.apply_along_axis(f, 0, u)
    u_next = ku + dt*fu
    soln.append(u_next)
  return np.asarray(soln)


if __name__ == "__main__":
  import matplotlib.pyplot as plt
  from PIL import Image
  from tqdm import tqdm

  S, I, R = "SIR"
  a, b = "ab"
  net = petrinet.PetriNet(
    species     = [S, I, R],
    transitions = [a, b],
    edges_in  = [(S,a), (I,a), (I,b)],
    edges_out = [(a,I), (a,I), (b,R)]
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

  f = lambda x: petrinet.time_dot(net, x=x, rates=[2.0, 1.0])
  soln = reaction_diffusion(x, dx, u0(x), lambda x: np.array(mass_action.time_dot(net, x, rates=[1.2, 0.1])), c=1, dt=0.1, num_steps=1000)

  # save result to gif
  frames = []
  fig = plt.figure()
  for i, u in tqdm(enumerate(soln)):
    fig.clf()
    p = fig.add_subplot(111)
    for i, s in enumerate(net.species):
      p.plot(x, u[i], label=s)
    fig.canvas.draw()
    image = Image.frombytes(
      mode="RGB",
      size=fig.canvas.get_width_height(),
      data=fig.canvas.tostring_rgb()
    )
    frames.append(image)

  frames[0].save(
    fp="figures/pde_demo.gif",
    format="GIF",
    append_images=frames,
    save_all=True,
    duration=50,
    loop=0
  )
