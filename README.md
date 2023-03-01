# petrinets

Python implementation of Petri nets.

The primary class is located in `petrinet.py` and mass action laws in `mass_action.py`. They are used in the following scripts:
- `demo_ode.py` - as an ordinary differential equation,
- `demo_sde.py` - as a stochastic differential equation,
- `demo_pde.py` - as a reaction diffusion PDE.

## Usage

The following example is the SIR epidemic model corresponding to the reaction network

$$ S + I \to 2I $$

$$ I \to R $$

```python
from petrinet import PetriNet

S, I, R = "SIR"
a, b = "ab"
net = PetriNet(
  species     = [S, I, R],
  transitions = [a, b],
  edges_in  = [(S,a), (I,a), (I,b)],
  edges_out = [(I,a), (I,a), (R,b)]
)
```

By assigning each reaction a rate, we can compute the time derivative using mass action kinetics:

```python
from mass_action import time_dot
x_dot = time_dot(net, x=[1.0, 2.0, 3.0], rates=[1.0, 1.2])
```

We can approximate a solution to the ODE using SciPy's solve_ivp, and plot the solution with Matplotlib.

```python
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

soln = solve_ivp(
  fun=lambda t, x: time_dot(net, x=x, rates=[2.0, 1.0]),
  t_span=[0.0, 10.0],
  y0=[0.99, 0.01, 0.00],
  method="RK45"
)

for i in range(len(soln.y)):
  plt.plot(soln.t, soln.y[i], label=i)
plt.legend()
plt.show()
```

![Time series plot of the SIR model](figures/ode_demo.png)
