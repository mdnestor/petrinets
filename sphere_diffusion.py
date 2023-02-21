import numpy as np
dx = 0.2
theta = np.arange(0, np.pi+dx, dx)
psi = np.arange(0, 2*np.pi+dx, dx)
# print(x)

all_pts = [(i,j) for i in range(len(theta)) for j in range(len(psi))]

all_pairs = [(p1, p2) for p1 in all_pts for p2 in all_pts]

# given a pair, how do i compute its distance?
def pair_dist(p1, p2):
  i1, j1 = p1
  i2, j2 = p2
  theta_1, psi_1 = theta[i1], psi[j1]
  theta_2, psi_2 = theta[i2], psi[j2]
  return np.arccos(np.cos(theta_1)*np.cos(theta_2) + np.sin(theta_1)*np.sin(theta_2)*np.cos(psi_1-psi_2))

# compute dist
dist = [pair_dist(p1,p2) for p1 in all_pts for p2 in all_pts]

n_points = len(all_pts)

dist = np.asarray(dist).reshape((n_points,n_points))

# now try to plot an arbitrary array:

U = np.zeros((len(theta), len(psi)), dtype=float) # order?


U[4,0] = 1.0

print(U.shape)

# take... input array

def laplacian(U):
  U_unwrapped = U.reshape(n_points)


  LU = np.zeros(n_points)
  for i in range(len(U_unwrapped)):
    p = [np.exp(-dist[i,j]*10) for j in range(n_points)]
    p = np.nan_to_num(p)
    p = np.array(p)
    p /= np.sum(p)
    LU += U_unwrapped[i] * p
  return LU.reshape(U.shape)



from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


fig = plt.figure()
ax = fig.add_subplot( 1, 1, 1, projection='3d')

u = psi
v = theta

# create the sphere surface
XX = 10 * np.outer( np.cos( u ), np.sin( v ) )
YY = 10 * np.outer( np.sin( u ), np.sin( v ) )
ZZ = 10 * np.outer( np.ones( np.size( u ) ), np.cos( v ) )

# define f pointwise
def f(u):
  return 2.0*u*(1-u)

dt = 0.1
def pde_step(U):
  return U + dt*(f(U) + laplacian(U))

U_next = pde_step(U)

myheatmap = U_next.T

# ~ ax.scatter( *zip( *pointList ), color='#dd00dd' )
ax.plot_surface( XX, YY,  ZZ, cstride=1, rstride=1, facecolors=cm.jet( myheatmap ) )
plt.savefig("sphere_plot.png")

# make into a gif
history = [U]

from tqdm import tqdm
import matplotlib.pyplot as plt

for _ in tqdm(range(50)):
  U = history[-1]
  history.append(pde_step(U))

from PIL import Image

frames = []

fig = plt.figure()

for i in tqdm(range(len(history))):
  U = history[i]
  plt.clf()
  ax = fig.add_subplot( 1, 1, 1, projection='3d')
  ax.set_axis_off()
  myheatmap = U.T

  ax.plot_surface( XX, YY,  ZZ, cstride=1, rstride=1, facecolors=cm.jet( myheatmap ) )
  fig.canvas.draw()
  image = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())

  frames.append(image)

frames[0].save("sphere_diffusion.gif", format="GIF", append_images=frames,
            save_all=True, duration=50, loop=0)
